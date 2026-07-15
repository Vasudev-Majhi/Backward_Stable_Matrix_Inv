"""Merged LU + direct-sensitivity DSL graph (one model.bin per circuit).

All numerical work for solving A_FF * v_F = -A_FP * v_P happens INSIDE the
transformer:
    - rhs_i tokens build b_i = sum_k g_ik * v_P_k via fetch from v_init.
    - fwd_i tokens fetch b_i from rhs_i and run forward sub: y_i = b_i - sum_{j<i} L[i,j] y_j.
    - bck_i tokens (reverse-ordered) run back sub: x_i = (y_i - sum_{j>i} U[i,j] x_j) / U[i,i].
    - readout fetches x_target (or v_init_target if target is fixed) and emits
      it via the standard quadratic-score v_k formula.

Build-time math is restricted to:
    1. Sparse conductance accumulation (build_partition).
    2. Doolittle LU on A_FF.

There is no numpy.linalg.solve at build time and no Python arithmetic at
inference time.
"""
from __future__ import annotations

import _path  # noqa: F401

import numpy as np

from parse import ParsedCircuit  # type: ignore
from cadj_reference import K_LEVELS, SCALE, V_STEP  # type: ignore
from transformer_vm.graph import core as _graph  # type: ignore
from transformer_vm.graph.core import (  # type: ignore
    Expression,
    InputDimension,
    ProgramGraph,
    auto_name,
    fetch,
    persist,
    reglu,
    reset_graph,
)

from lu_direct_reference import build_partition
from lu_factor import doolittle

H_OUT = 1e5
# M_SOURCES_HARD_CAP is the largest n_fixed any circuit in the dataset can have
# (used only as a sanity check). The actual rhs source-bank width is set
# per-circuit to n_fixed -- this matters for MILP cost: declaring the bank
# at a static 10 always inflates d_model by ~30 input dims even when
# n_fixed=2, multiplying MILP runtime several-fold for small circuits.
M_SOURCES_HARD_CAP = 10


def _afp_row(A_FP_neg: np.ndarray, i: int) -> list[tuple[int, float]]:
    """Return [(slot_k_in_fixed, g_ik), ...] for non-zero entries of -A_FP[i,:]."""
    pairs: list[tuple[int, float]] = []
    for k in range(A_FP_neg.shape[1]):
        g = float(A_FP_neg[i, k])
        if abs(g) > 1e-15:
            pairs.append((k, g))
    return pairs


def build_lu_direct_graph(
    pc: ParsedCircuit,
    target_node: int,
    v_step: int | None = None,
    k_levels: int | None = None,
):
    """Return (input_tokens, output_tokens, meta).

    meta carries layout positions, n_free, free/fixed indices, and the
    Dimension handles needed by the build pipeline to expose slot indices
    in the .slots.json sidecar.
    """
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes

    # ---- Build-time linear-algebra (sparse only; no linalg.solve) ---------
    free, fixed, A_FF, A_FP = build_partition(pc)
    n_free = len(free)
    n_fixed = len(fixed)
    if n_fixed > M_SOURCES_HARD_CAP:
        raise ValueError(
            f"n_fixed={n_fixed} exceeds M_SOURCES_HARD_CAP={M_SOURCES_HARD_CAP}"
        )
    # Per-circuit source-bank width: declare exactly n_fixed slots (or 1 when
    # n_fixed=0 to keep the rhs loop non-empty). Reduces MILP cost dramatically
    # vs. always declaring M_SOURCES_HARD_CAP slots.
    M = max(n_fixed, 1)

    target_is_fixed = pc.is_fixed[target_node]
    if not target_is_fixed:
        free_map = {n: i for i, n in enumerate(free)}
        ti = free_map[target_node]
    else:
        ti = -1

    if n_free > 0 and not target_is_fixed:
        # A_FP carries -A_FP[i,k] = -(-1/R) = +g_ik for parallel resistors.
        # We want -A_FP @ v_P = sum_k g_ik * v_P_k.  So afp = -A_FP.
        afp = -A_FP
        L, U = doolittle(A_FF)
    else:
        afp = -A_FP if n_free > 0 else np.zeros((0, n_fixed))
        L = np.eye(max(n_free, 1))
        U = np.eye(max(n_free, 1))

    # Layout positions (must match lu_direct_tokenize.tokenize_lu_direct).
    pos_rhs_base = 1 + 2 * N
    pos_fwd_base = pos_rhs_base + n_free
    pos_bck_base = pos_fwd_base + n_free
    pos_readout = pos_bck_base + n_free

    # ---- InputDimensions -------------------------------------------------
    # Token-type flags
    is_v_emission  = InputDimension("is_v_emission")
    is_skip_tok    = InputDimension("is_skip_tok")
    is_rhs_tok     = InputDimension("is_rhs_tok")
    is_fwd_tok     = InputDimension("is_fwd_tok")
    is_bck_tok     = InputDimension("is_bck_tok")
    is_readout_tok = InputDimension("is_readout_tok")

    # Universal value slot (read by every fetch; runner-injected after
    # rhs/fwd/bck steps via V-cache patch).
    v_value_slot = InputDimension("v_value_slot")
    x_value_slot = InputDimension("x_value_slot")
    y_input_slot = InputDimension("y_input_slot")
    node_id      = InputDimension("node_id")

    # LU diagonals (per-token).
    inv_l_pos_dim = InputDimension("inv_l_pos")  # +1/L[i,i] on fwd_i (=1 always)
    inv_l_neg_dim = InputDimension("inv_l_neg")
    inv_u_pos_dim = InputDimension("inv_u_pos")  # +1/U[i,i] on bck_i
    inv_u_neg_dim = InputDimension("inv_u_neg")

    # RHS source bank — fetches v_init_k via fixed offset.
    afp_off   = [InputDimension(f"afp_off_{k}")   for k in range(M)]
    afp_w_pos = [InputDimension(f"afp_w_pos_{k}") for k in range(M)]
    afp_w_neg = [InputDimension(f"afp_w_neg_{k}") for k in range(M)]
    afp_valid = [InputDimension(f"afp_valid_{k}") for k in range(M)]

    # Forward-sub neighbour bank (fwd_j with j<i)  — size n_free-1 worst case.
    D_MAX_FWD = max(n_free - 1, 1)
    D_MAX_BCK = max(n_free - 1, 1)
    lwd_off   = [InputDimension(f"lwd_off_{k}")   for k in range(D_MAX_FWD)]
    lwd_w_pos = [InputDimension(f"lwd_w_pos_{k}") for k in range(D_MAX_FWD)]
    lwd_w_neg = [InputDimension(f"lwd_w_neg_{k}") for k in range(D_MAX_FWD)]
    lwd_valid = [InputDimension(f"lwd_valid_{k}") for k in range(D_MAX_FWD)]
    uwd_off   = [InputDimension(f"uwd_off_{k}")   for k in range(D_MAX_BCK)]
    uwd_w_pos = [InputDimension(f"uwd_w_pos_{k}") for k in range(D_MAX_BCK)]
    uwd_w_neg = [InputDimension(f"uwd_w_neg_{k}") for k in range(D_MAX_BCK)]
    uwd_valid = [InputDimension(f"uwd_valid_{k}") for k in range(D_MAX_BCK)]

    # Readout source bank — single offset slot. Build-time picks whether the
    # fetch reads x_value_slot (free target -> bck_ti) or v_value_slot (fixed
    # target -> v_init_target). Mirrors direct_interpreter's M_SOURCES_MAX=1 path.
    rd_off = InputDimension("rd_off")
    is_src_rd = InputDimension("is_src_rd")

    # ---- DSL: rhs accumulator (computes b_i on rhs_i tokens) -------------
    b_acc = Expression()
    for k in range(M):
        v_src_k = fetch(
            value=v_value_slot,
            query=position + afp_off[k],
            key=position,
            clear_key=1 - is_v_emission,
            tie_break="latest",
        )
        gated = reglu(v_src_k, afp_valid[k])
        pos_contrib = reglu(gated, afp_w_pos[k])
        neg_contrib = reglu(gated, afp_w_neg[k])
        b_acc = b_acc + pos_contrib - neg_contrib
    b_value = persist(reglu(b_acc, is_rhs_tok), name="b_value")

    # ---- DSL: forward-sub neighbour sum (on fwd_i) -----------------------
    sum_term_fwd = Expression()
    for k in range(D_MAX_FWD):
        fetch_lwd_k = fetch(
            value=x_value_slot,
            query=position + lwd_off[k],
            key=position,
            clear_key=1 - is_fwd_tok,
            tie_break="latest",
        )
        pos_contrib = reglu(fetch_lwd_k, lwd_w_pos[k])
        neg_contrib = reglu(fetch_lwd_k, lwd_w_neg[k])
        pos_gated = reglu(pos_contrib, lwd_valid[k])
        neg_gated = reglu(neg_contrib, lwd_valid[k])
        sum_term_fwd = sum_term_fwd + pos_gated - neg_gated

    # ---- DSL: back-sub neighbour sum (on bck_i) --------------------------
    sum_term_bck = Expression()
    for k in range(D_MAX_BCK):
        fetch_uwd_k = fetch(
            value=x_value_slot,
            query=position + uwd_off[k],
            key=position,
            clear_key=1 - is_bck_tok,
            tie_break="latest",
        )
        pos_contrib = reglu(fetch_uwd_k, uwd_w_pos[k])
        neg_contrib = reglu(fetch_uwd_k, uwd_w_neg[k])
        pos_gated = reglu(pos_contrib, uwd_valid[k])
        neg_gated = reglu(neg_contrib, uwd_valid[k])
        sum_term_bck = sum_term_bck + pos_gated - neg_gated

    # ---- DSL: b-fetch for fwd_i (replaces b_value_slot injection) --------
    # rhs_i lives at pos_rhs_base+i; fwd_i at pos_fwd_base+i. Offset = -n_free.
    b_fetched = fetch(
        value=x_value_slot,
        query=position + (-n_free),
        key=position,
        clear_key=1 - is_rhs_tok,
        tie_break="latest",
    )
    b_pos_term = reglu(b_fetched, inv_l_pos_dim)
    b_neg_term = reglu(b_fetched, inv_l_neg_dim)
    b_pos_term = reglu(b_pos_term, is_fwd_tok)
    b_neg_term = reglu(b_neg_term, is_fwd_tok)
    b_term = b_pos_term - b_neg_term

    # ---- DSL: y-injection for bck_i (runner-injected via y_input_slot) ---
    y_pos_term = reglu(y_input_slot, inv_u_pos_dim)
    y_neg_term = reglu(y_input_slot, inv_u_neg_dim)
    y_pos_term = reglu(y_pos_term, is_bck_tok)
    y_neg_term = reglu(y_neg_term, is_bck_tok)
    y_term = y_pos_term - y_neg_term

    # x_new persists once for both fwd_i (= y_i) and bck_i (= x_i).
    x_new = persist(b_term + sum_term_fwd + y_term + sum_term_bck, name="x_new")

    # ---- DSL: readout fetches x_target or v_init_target ------------------
    # Single per-readout fetch, branching on target_is_fixed at build time.
    if target_is_fixed:
        target_fetch = fetch(
            value=v_value_slot,
            query=position + rd_off,
            key=position,
            clear_key=1 - is_v_emission,
            tie_break="latest",
        )
    else:
        target_fetch = fetch(
            value=x_value_slot,
            query=position + rd_off,
            key=position,
            clear_key=1 - is_bck_tok,
            tie_break="latest",
        )
    # is_src_rd gates inactive readouts to zero; v_score_source is the only
    # persist needed before the quadratic v_k score.
    v_direct_acc = reglu(target_fetch, is_src_rd)
    v_score_source = persist(reglu(v_direct_acc, is_readout_tok), name="v_score_source")
    emit_v_gate = persist(is_readout_tok, name="emit_v_gate")

    # ---- input_tokens ----------------------------------------------------
    input_tokens: dict[str, Expression] = {}
    input_tokens["start"] = Expression()
    input_tokens["skip"]  = 1 * is_skip_tok
    input_tokens["halt"]  = Expression()

    for k in range(k_levels):
        input_tokens[f"v_{k}"] = 1 * is_v_emission + (k * v_step) * v_value_slot

    # rhs_i: compute b_i = sum_k g_ik * v_P_k.
    # afp_off[slot] = (2*fixed[slot]+2) - (pos_rhs_base+i)
    for i in range(n_free):
        emb = 1 * is_rhs_tok + free[i] * node_id
        my_pos = pos_rhs_base + i
        for slot in range(M):
            if slot < n_fixed and not target_is_fixed:
                g_ik = float(afp[i, slot])
                v_init_k_pos = 2 * fixed[slot] + 2
                off = v_init_k_pos - my_pos  # always negative
                w_pos = max(0.0, +g_ik)
                w_neg = max(0.0, -g_ik)
                emb = emb + off * afp_off[slot]
                emb = emb + w_pos * afp_w_pos[slot]
                emb = emb + w_neg * afp_w_neg[slot]
                emb = emb + 1 * afp_valid[slot]
            else:
                emb = emb + (-10 * N - 1000) * afp_off[slot]
                # afp_valid[slot] = 0
        input_tokens[f"rhs_{i}"] = emb

    # fwd_i: forward-sub for row i.
    for i in range(n_free):
        if target_is_fixed:
            inv_lii = 1.0
            row_others: list[tuple[int, float]] = []
        else:
            inv_lii = 1.0 / L[i, i]   # = 1 for Doolittle
            row_others = [(j, -L[i, j] * inv_lii) for j in range(i)]

        emb = (
            1 * is_fwd_tok
            + free[i] * node_id
            + max(0.0, +inv_lii) * inv_l_pos_dim
            + max(0.0, -inv_lii) * inv_l_neg_dim
        )
        for slot, (j, coef) in enumerate(row_others):
            # fwd_j at pos_fwd_base+j; offset = j - i (negative).
            off = j - i
            w_pos = max(0.0, +coef)
            w_neg = max(0.0, -coef)
            emb = emb + off * lwd_off[slot]
            emb = emb + w_pos * lwd_w_pos[slot]
            emb = emb + w_neg * lwd_w_neg[slot]
            emb = emb + 1 * lwd_valid[slot]
        for slot in range(len(row_others), D_MAX_FWD):
            emb = emb + (-10 * N - 1000) * lwd_off[slot]
        input_tokens[f"fwd_{i}"] = emb

    # bck_i: back-sub for row i.
    for i in range(n_free):
        if target_is_fixed:
            inv_uii = 1.0
            row_others = []
        else:
            inv_uii = 1.0 / U[i, i]
            row_others = [(j, -U[i, j] * inv_uii) for j in range(i + 1, n_free)]

        emb = (
            1 * is_bck_tok
            + free[i] * node_id
            + max(0.0, +inv_uii) * inv_u_pos_dim
            + max(0.0, -inv_uii) * inv_u_neg_dim
        )
        for slot, (j, coef) in enumerate(row_others):
            # bck_j at pos_bck_base + (n_free-1-j); bck_i at pos_bck_base + (n_free-1-i).
            # Offset = (n_free-1-j) - (n_free-1-i) = i - j (negative since j > i).
            off = i - j
            w_pos = max(0.0, +coef)
            w_neg = max(0.0, -coef)
            emb = emb + off * uwd_off[slot]
            emb = emb + w_pos * uwd_w_pos[slot]
            emb = emb + w_neg * uwd_w_neg[slot]
            emb = emb + 1 * uwd_valid[slot]
        for slot in range(len(row_others), D_MAX_BCK):
            emb = emb + (-10 * N - 1000) * uwd_off[slot]
        input_tokens[f"bck_{i}"] = emb

    # readout: one fetch, with rd_off / is_src_rd carrying the active offset.
    readout_emb = 1 * is_readout_tok + 1 * is_src_rd
    if target_is_fixed:
        v_init_target_pos = 2 * target_node + 2
        readout_emb = readout_emb + (v_init_target_pos - pos_readout) * rd_off
    else:
        bck_ti_pos = pos_bck_base + (n_free - 1 - ti)
        readout_emb = readout_emb + (bck_ti_pos - pos_readout) * rd_off  # = -(1+ti)
    input_tokens["readout"] = readout_emb

    # one=1 on every non-start token (Project A/B convention).
    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens: quadratic v_k scoring on readout -----------------
    output_tokens: dict[str, Expression] = {tok: Expression() for tok in input_tokens}
    for k in range(k_levels):
        vk = k * v_step
        output_tokens[f"v_{k}"] = (
            H_OUT * emit_v_gate
            + (2 * vk) * v_score_source
            + Expression({one: -(vk * vk)})
        )
    output_tokens["halt"] = Expression({one: 0})

    # Reference x_new from fwd/bck output tokens so MILP reserves a slot.
    for tok in output_tokens:
        if tok.startswith("fwd_") or tok.startswith("bck_"):
            output_tokens[tok] = output_tokens[tok] + x_new
        if tok.startswith("rhs_"):
            output_tokens[tok] = output_tokens[tok] + b_value

    meta = {
        "N": N,
        "n_free": n_free,
        "n_fixed": n_fixed,
        "target_node": target_node,
        "target_is_fixed": target_is_fixed,
        "target_free_index": ti,
        "free": free,
        "fixed": fixed,
        "pos_rhs_base": pos_rhs_base,
        "pos_fwd_base": pos_fwd_base,
        "pos_bck_base": pos_bck_base,
        "pos_readout": pos_readout,
        "x_value_slot_dim": x_value_slot,
        "v_value_slot_dim": v_value_slot,
        "y_input_slot_dim": y_input_slot,
        "is_v_emission_dim": is_v_emission,
        "b_value_expr": b_value,
        "x_new_expr": x_new,
        "v_score_source_expr": v_score_source,
        "emit_v_gate_expr": emit_v_gate,
        "L": L,
        "U": U,
    }
    auto_name(locals())
    return input_tokens, output_tokens, meta


class LuDirectCircuitMachine:
    def __init__(
        self,
        parsed: ParsedCircuit,
        target_node: int,
        v_step: int | None = None,
        k_levels: int | None = None,
    ):
        self.parsed = parsed
        self.target_node = target_node
        self.v_step = v_step
        self.k_levels = k_levels

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_lu_direct_graph(
            self.parsed, self.target_node,
            v_step=self.v_step, k_levels=self.k_levels,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
