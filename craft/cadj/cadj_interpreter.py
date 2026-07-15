"""Per-circuit Chebyshev-Accelerated Discrete Jacobi (CADJ) DSL graph.

Architecture clone of `rbsor_interpreter.py` with three deltas:
  1. Two self-history fetches (v_old at offset -2N, v_oldold at offset -4N) —
     RB-SOR had only one (v_old at -2N).
  2. No graph coloring — natural node order 0..N-1 in each iteration block.
     Per-neighbor offset is the standard Jacobi formula:
        off_k = -2N + 2*(nbr_id - n) + 1
  3. Three-term Chebyshev recurrence in place of SOR's 2-term combo:
        v_cadj_free = alpha * v_jacobi + beta * v_old + gamma * v_oldold
     Per-iteration coefficients (alpha_k, beta_k, gamma_k) are baked into
     the per-(node, phase) update token's embedding via three new
     InputDimensions: coef_alpha, coef_beta, coef_gamma_neg. Coefficients are
     pre-scaled by COEF_SCALE=10000 so they fit as integers in the
     embedding; the (1.0 / COEF_SCALE) factor is folded into the persist
     expression (same pattern as Jacobi's `sum_wv * (1.0/SCALE)`).
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

from parse import ParsedCircuit
from transformer_vm.graph import core as _graph
from transformer_vm.graph.core import (
    Expression,
    InputDimension,
    ProgramGraph,
    auto_name,
    fetch,
    persist,
    reglu,
    reset_graph,
)

from cadj_reference import K_LEVELS, SCALE, V_STEP, chebyshev_coefficients
from cadj_tokenize import NUM_PHASES, coefficients_per_phase

D_MAX = 8
H_OUT = 1e5
COEF_SCALE = 10000   # integer scaling for Chebyshev coefficients in token embeddings


def build_cadj_circuit_graph(
    pc: ParsedCircuit,
    target_node: int,
    T: int,
    lam_min: float,
    lam_max: float,
    v_step: int | None = None,
    k_levels: int | None = None,
):
    """Return (input_tokens, output_tokens, meta). Assumes reset_graph() called."""
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes

    # Compute per-phase Chebyshev coefficients (averages within each phase).
    a_full, b_full, g_full = chebyshev_coefficients(lam_min, lam_max, T)
    a_phase, b_phase, g_phase = coefficients_per_phase(a_full, b_full, g_full)

    # ---- Input dimensions -----------------------------------------------
    is_update_tok   = InputDimension("is_update_tok")
    is_v_emission   = InputDimension("is_v_emission")
    is_readout_tok  = InputDimension("is_readout_tok")
    is_skip_tok     = InputDimension("is_skip_tok")

    v_value_slot = InputDimension("v_value_slot")

    node_id      = InputDimension("node_id")
    is_fixed     = InputDimension("is_fixed")
    init_voltage = InputDimension("init_voltage")
    iter_phase   = InputDimension("iter_phase")   # debug-only

    # NEW: per-token Chebyshev coefficients (pre-scaled by COEF_SCALE).
    # All three are stored as NON-NEGATIVE gates so they can drive reglu.
    # alpha_k >= 0 and beta_k >= 0 always (in the Chebyshev branch);
    # gamma_k <= 0 in the Chebyshev branch, so we store |gamma_k| and apply
    # the -1 sign at build time. Plain-Jacobi branch (kappa < 2) uses
    # alpha=1, beta=0, gamma=0 -> all three remain non-negative.
    coef_alpha     = InputDimension("coef_alpha")
    coef_beta      = InputDimension("coef_beta")
    coef_gamma_neg = InputDimension("coef_gamma_neg")

    nbr_off   = [InputDimension(f"nbr_off_{k}")   for k in range(D_MAX)]
    nbr_w     = [InputDimension(f"nbr_w_{k}")      for k in range(D_MAX)]
    nbr_valid = [InputDimension(f"nbr_valid_{k}")  for k in range(D_MAX)]

    readout_offset = InputDimension("readout_offset")

    # ---- DSL: update-token compute --------------------------------------
    # 6-neighbor Jacobi sum (unchanged from existing Jacobi/RB-SOR).
    msgs: list[Expression] = []
    for k in range(D_MAX):
        v_to_k = fetch(
            value=v_value_slot,
            query=position + nbr_off[k],
            key=position,
            clear_key=1 - is_v_emission,
            tie_break="latest",
        )
        msg_k = reglu(v_to_k, nbr_w[k])
        msg_k_gated = reglu(msg_k, nbr_valid[k])
        msgs.append(msg_k_gated)

    sum_wv = Expression()
    for m in msgs:
        sum_wv = sum_wv + m
    sum_wv = persist(sum_wv, name="sum_wv")
    v_jacobi = persist(sum_wv * (1.0 / SCALE), name="v_jacobi")

    # NEW: two self-history fetches.
    # v_old: previous iteration's v emission for this node (offset -2N).
    v_old_raw = fetch(
        value=v_value_slot,
        query=position + (-2 * N),
        key=position,
        clear_key=1 - is_v_emission,
        tie_break="latest",
    )
    v_old = persist(v_old_raw, name="v_old")
    # v_oldold: two iterations back (offset -4N).
    v_oldold_raw = fetch(
        value=v_value_slot,
        query=position + (-4 * N),
        key=position,
        clear_key=1 - is_v_emission,
        tie_break="latest",
    )
    v_oldold = persist(v_oldold_raw, name="v_oldold")

    # Chebyshev recurrence: alpha*v_jacobi + beta*v_old + gamma*v_oldold.
    # The DSL doesn't support runtime*runtime multiply directly; we use
    # reglu(value, gate) which computes value * max(0, gate). All three
    # coefficient gates are non-negative (see InputDimension comment above);
    # the negative gamma sign is applied as a build-time -1 scalar.
    # COEF_SCALE divisor is folded into the persist (same pattern as
    # `sum_wv * (1.0/SCALE)` for v_jacobi).
    v_cadj_free = persist(
        (1.0 / COEF_SCALE) * (
            reglu(v_jacobi, coef_alpha)
            + reglu(v_old, coef_beta)
            + (-1.0) * reglu(v_oldold, coef_gamma_neg)
        ),
        name="v_cadj_free",
    )

    # Fixed-node gate. reglu(v_cadj_free, 1-is_fixed) clips negatives to 0
    # for free nodes — mirrors the reference's _quantize_to_vk(..., min=0).
    v_new = persist(
        reglu(init_voltage, is_fixed) + reglu(v_cadj_free, 1 - is_fixed),
        name="v_new",
    )

    # ---- DSL: readout-token compute -------------------------------------
    target_v = fetch(
        value=v_value_slot,
        query=position + readout_offset,
        key=position,
        clear_key=1 - is_v_emission,
        tie_break="latest",
    )
    target_v_persist = persist(target_v, name="target_v")

    v_score_source = persist(
        reglu(v_new, is_update_tok) + reglu(target_v_persist, is_readout_tok),
        name="v_score_source",
    )
    emit_v_gate = persist(is_update_tok + is_readout_tok, name="emit_v_gate")

    # ---- input_tokens ---------------------------------------------------
    input_tokens: dict[str, Expression] = {}
    input_tokens["start"] = Expression()
    input_tokens["skip"]  = 1 * is_skip_tok
    input_tokens["halt"]  = Expression()

    for k in range(k_levels):
        input_tokens[f"v_{k}"] = 1 * is_v_emission + (k * v_step) * v_value_slot

    # Update tokens: one per (node, phase). Each phase carries that phase's
    # averaged Chebyshev coefficients (pre-scaled to integers).
    for n in range(N):
        fixed_flag = 1 if pc.is_fixed[n] else 0
        v0 = int(round(pc.fixed_voltage[n] * SCALE))

        out = pc.out_edges_norm[n]
        nbr_pairs = list(out)[:D_MAX]

        # Per-neighbor-slot fields. Standard Jacobi offset (no coloring).
        per_slot_off: list[int] = []
        per_slot_w:   list[int] = []
        per_slot_valid: list[int] = []
        for k in range(D_MAX):
            if k < len(nbr_pairs):
                nbr_id, w = nbr_pairs[k]
                off = -2 * N + 2 * (nbr_id - n) + 1
                per_slot_off.append(off)
                per_slot_w.append(w)
                per_slot_valid.append(1)
            else:
                per_slot_off.append(0)
                per_slot_w.append(0)
                per_slot_valid.append(0)

        for phase in range(NUM_PHASES):
            a_int = int(round(max(a_phase[phase], 0.0) * COEF_SCALE))
            b_int = int(round(max(b_phase[phase], 0.0) * COEF_SCALE))
            # gamma is <=0 in Chebyshev branch and ==0 in Jacobi branch;
            # store its magnitude so it serves as a non-negative reglu gate.
            g_neg_int = int(round(max(-g_phase[phase], 0.0) * COEF_SCALE))

            emb = (
                1 * is_update_tok
                + n * node_id
                + fixed_flag * is_fixed
                + v0 * init_voltage
                + phase * iter_phase
                + a_int     * coef_alpha
                + b_int     * coef_beta
                + g_neg_int * coef_gamma_neg
            )
            for k in range(D_MAX):
                if per_slot_valid[k]:
                    emb = emb + per_slot_off[k]   * nbr_off[k]
                    emb = emb + per_slot_w[k]     * nbr_w[k]
                    emb = emb + 1                 * nbr_valid[k]
            input_tokens[f"up_{n}_t{phase}"] = emb

    # Readout: position offset to target's final v-emission.
    # The final iteration's v-emission for target_node is at:
    #     pos = 2N + 2 + 2*((T-1)*N + target_node)
    # The readout token sits at:
    #     pos = 2N + 1 + 2*T*N
    # so:
    #     ro_off = (2N + 2 + 2*((T-1)*N + target)) - (2N + 1 + 2*T*N)
    #            = 2*target - 2N + 1
    ro_off = 2 * target_node - 2 * N + 1
    input_tokens["readout"] = (
        1 * is_readout_tok + ro_off * readout_offset
    )

    # Every non-start token carries one=1.
    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens: quadratic scoring for v-token emission ----------
    output_tokens: dict[str, Expression] = {tok: Expression() for tok in input_tokens}
    for k in range(k_levels):
        vk = k * v_step
        output_tokens[f"v_{k}"] = (
            H_OUT * emit_v_gate
            + (2 * vk) * v_score_source
            + Expression({one: -(vk * vk)})
        )
    output_tokens["halt"] = Expression({one: 0})

    auto_name(locals())
    meta = {
        "N": N,
        "T": T,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "num_phases": NUM_PHASES,
        "alphas_per_phase": a_phase,
        "betas_per_phase": b_phase,
        "gammas_per_phase": g_phase,
        "target_node": target_node,
        "target_v_expr": target_v_persist,
        "v_new_expr": v_new,
    }
    return input_tokens, output_tokens, meta


class CADJCircuitMachine:
    def __init__(
        self,
        parsed: ParsedCircuit,
        target_node: int,
        T: int,
        lam_min: float,
        lam_max: float,
        v_step: int | None = None,
        k_levels: int | None = None,
    ):
        self.parsed = parsed
        self.target_node = target_node
        self.T = T
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.v_step = v_step
        self.k_levels = k_levels

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_cadj_circuit_graph(
            self.parsed, self.target_node, self.T,
            self.lam_min, self.lam_max,
            v_step=self.v_step, k_levels=self.k_levels,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
