"""Per-circuit Forward Red-Black SOR DSL graph.

Architecture clone of `interpreter.py` (Jacobi) with three deltas:
  1. Per-neighbor offset is color-aware:
     - same color (conflict edge):           prev iter, offset = -2N + 2*(cidx(nbr) - cidx(n)) + 1
     - red->black (n is red, nbr is black):  prev iter, offset = -2N + 2*(cidx(nbr) - cidx(n)) + 1
     - black->red (n is black, nbr is red):  CURRENT iter, offset = 2*(cidx(nbr) - cidx(n)) + 1
     where cidx(n) = position of n in red_order ++ black_order.
  2. Extra fetch per update token: v_old at constant offset -2N (own previous v_k).
  3. SOR combination: v_sor_free = (1-omega)*v_old + omega*v_jacobi (single persist).

The token sequence uses the same red ++ black per-iteration block (see
rbsor_tokenize.py) so all offsets are iteration-independent.
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

from coloring import color_idx_map
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

from rbsor_reference import K_LEVELS, SCALE, V_STEP

D_MAX = 6
H_OUT = 1e5


def build_rbsor_circuit_graph(
    pc: ParsedCircuit,
    target_node: int,
    T: int,
    omega: float,
    red_order: list[int],
    black_order: list[int],
    v_step: int | None = None,
    k_levels: int | None = None,
):
    """Return (input_tokens, output_tokens, meta). Assumes reset_graph() was called."""
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes
    n_red = len(red_order)
    if N != n_red + len(black_order):
        raise ValueError("coloring partition does not cover all nodes")

    cidx = color_idx_map(red_order, black_order)         # node id -> color_idx in [0,N)
    red_set = set(red_order)

    # ---- Input dimensions -----------------------------------------------
    is_update_tok   = InputDimension("is_update_tok")
    is_v_emission   = InputDimension("is_v_emission")
    is_readout_tok  = InputDimension("is_readout_tok")
    is_skip_tok     = InputDimension("is_skip_tok")

    # Debug-only: doesn't gate anything; useful for tracing hidden states.
    is_red_update   = InputDimension("is_red_update")
    is_black_update = InputDimension("is_black_update")

    v_value_slot = InputDimension("v_value_slot")

    node_id       = InputDimension("node_id")
    is_fixed      = InputDimension("is_fixed")
    init_voltage  = InputDimension("init_voltage")
    iter_parity   = InputDimension("iter_parity")

    nbr_off   = [InputDimension(f"nbr_off_{k}")    for k in range(D_MAX)]
    nbr_w     = [InputDimension(f"nbr_w_{k}")       for k in range(D_MAX)]
    nbr_valid = [InputDimension(f"nbr_valid_{k}")   for k in range(D_MAX)]

    readout_offset = InputDimension("readout_offset")

    # ---- DSL: update-token compute --------------------------------------
    # Same 6-neighbor fetch+reglu+sum pattern as Jacobi. Color-aware offset
    # is baked into the per-token embedding (see input_tokens loop below).
    msgs: list[Expression] = []
    for k in range(D_MAX):
        v_to_k = fetch(
            value=v_value_slot,
            query=position + nbr_off[k],
            key=position,
            clear_key=1 - is_v_emission,
            tie_break="latest",
        )
        msg_k = reglu(v_to_k, nbr_w[k])             # w_k * v_to_k  (w_k >= 0)
        msg_k_gated = reglu(msg_k, nbr_valid[k])
        msgs.append(msg_k_gated)

    sum_wv = Expression()
    for m in msgs:
        sum_wv = sum_wv + m
    sum_wv = persist(sum_wv, name="sum_wv")
    v_jacobi = persist(sum_wv * (1.0 / SCALE), name="v_jacobi")

    # NEW: fetch own previous-iteration v at constant offset -2N. Same key
    # filter as neighbor fetches (clear_key = 1 - is_v_emission).
    v_old_raw = fetch(
        value=v_value_slot,
        query=position + (-2 * N),
        key=position,
        clear_key=1 - is_v_emission,
        tie_break="latest",
    )
    v_old = persist(v_old_raw, name="v_old")

    # SOR combination as a single Expression with build-time constant scalars.
    # (1-omega) is negative for omega > 1; that's fine — Expression supports
    # signed coefficients on persisted dims (interpreter.py:131 already uses
    # an arbitrary scalar multiply with `sum_wv * (1.0/SCALE)`).
    v_sor_free = persist(
        (1.0 - omega) * v_old + omega * v_jacobi,
        name="v_sor_free",
    )

    # Fixed-node gate. reglu(v_sor_free, 1-is_fixed) clips negatives to 0
    # for free nodes — mirrors max(0, sor) in the reference.
    v_new = persist(
        reglu(init_voltage, is_fixed) + reglu(v_sor_free, 1 - is_fixed),
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

    # Update tokens: one per (node, parity). Name encodes color ("red"/"blk").
    for n in range(N):
        is_red = n in red_set
        prefix = "red" if is_red else "blk"
        fixed_flag = 1 if pc.is_fixed[n] else 0
        v0 = int(round(pc.fixed_voltage[n] * SCALE))

        out = pc.out_edges_norm[n]
        nbr_pairs = list(out)[:D_MAX]   # (nbr_id, w_norm_int) pairs

        # Color-aware per-neighbor offsets:
        #   black updater fetching a red neighbor -> CURRENT iter (offset = 2*(cidx_nbr - cidx_n) + 1)
        #   all other cases                       -> PREV iter    (offset = -2N + 2*(cidx_nbr - cidx_n) + 1)
        per_slot_off: list[int] = []
        per_slot_w:   list[int] = []
        per_slot_valid: list[int] = []
        for k in range(D_MAX):
            if k < len(nbr_pairs):
                nbr_id, w = nbr_pairs[k]
                delta = 2 * (cidx[nbr_id] - cidx[n]) + 1
                if (not is_red) and (nbr_id in red_set):
                    off = delta              # Case C: black -> red, current iter
                else:
                    off = -2 * N + delta     # Case A or B: prev iter
                per_slot_off.append(off)
                per_slot_w.append(w)
                per_slot_valid.append(1)
            else:
                per_slot_off.append(0)
                per_slot_w.append(0)
                per_slot_valid.append(0)

        for p in (0, 1):
            emb = (
                1 * is_update_tok
                + n * node_id
                + fixed_flag * is_fixed
                + v0 * init_voltage
                + p * iter_parity
                + (1 if is_red else 0) * is_red_update
                + (0 if is_red else 1) * is_black_update
            )
            for k in range(D_MAX):
                if per_slot_valid[k]:
                    emb = emb + per_slot_off[k]   * nbr_off[k]
                    emb = emb + per_slot_w[k]     * nbr_w[k]
                    emb = emb + 1                 * nbr_valid[k]
            input_tokens[f"up_{prefix}_{n}_p{p}"] = emb

    # Readout: position offset to the target's final v-emission.
    # The final iteration's v-emission for `target_node` is at
    #     pos = 2N + 2 + 2*((T-1)*N + cidx[target_node])
    # The readout token sits at
    #     pos = 2N + 1 + 2*T*N
    # so:
    #     ro_off = (2N + 2 + 2*((T-1)*N + cidx[target])) - (2N + 1 + 2*T*N)
    #            = 1 + 2*cidx[target] - 2N
    #            = 2*cidx[target] - 2N + 1
    ro_off = 2 * cidx[target_node] - 2 * N + 1
    input_tokens["readout"] = (
        1 * is_readout_tok + ro_off * readout_offset
    )

    # Every non-start token carries one=1 (matches interpreter.py).
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
        "omega": omega,
        "n_red": n_red,
        "n_black": len(black_order),
        "target_node": target_node,
        "target_v_expr": target_v_persist,
        "v_new_expr": v_new,
    }
    return input_tokens, output_tokens, meta


class RBSORCircuitMachine:
    def __init__(
        self,
        parsed: ParsedCircuit,
        target_node: int,
        T: int,
        omega: float,
        red_order: list[int],
        black_order: list[int],
        v_step: int | None = None,
        k_levels: int | None = None,
    ):
        self.parsed = parsed
        self.target_node = target_node
        self.T = T
        self.omega = omega
        self.red_order = red_order
        self.black_order = black_order
        self.v_step = v_step
        self.k_levels = k_levels

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_rbsor_circuit_graph(
            self.parsed, self.target_node, self.T, self.omega,
            self.red_order, self.black_order,
            v_step=self.v_step, k_levels=self.k_levels,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
