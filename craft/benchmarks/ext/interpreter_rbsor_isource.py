"""RB-SOR DSL graph WITH current-source (I) support.

Mirrors `rbsor/rbsor_interpreter.py` exactly except for the same I-source
addition as `interpreter_isource.py`:

    v_jacobi  =  sum_wv * (1/SCALE)  +  i_inj_slot
    v_sor_free = (1 - omega) * v_old + omega * v_jacobi   # unchanged

When pc.i_inj_norm is all zeros, the produced graph is behaviorally identical
to the standard RB-SOR graph (with one extra zero-coefficient input dim).
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

from coloring import color_idx_map
from rbsor_interpreter import D_MAX, H_OUT
from rbsor_reference import K_LEVELS, SCALE, V_STEP
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

from benchmarks.ext.parse_isource import ParsedCircuitWithI


def build_rbsor_circuit_graph_isource(
    pc: ParsedCircuitWithI,
    target_node: int,
    T: int,
    omega: float,
    red_order: list[int],
    black_order: list[int],
    v_step: int | None = None,
    k_levels: int | None = None,
):
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes
    n_red = len(red_order)
    if N != n_red + len(black_order):
        raise ValueError("coloring partition does not cover all nodes")

    cidx = color_idx_map(red_order, black_order)
    red_set = set(red_order)

    # ---- Input dimensions -----------------------------------------------
    is_update_tok   = InputDimension("is_update_tok")
    is_v_emission   = InputDimension("is_v_emission")
    is_readout_tok  = InputDimension("is_readout_tok")
    is_skip_tok     = InputDimension("is_skip_tok")
    is_red_update   = InputDimension("is_red_update")
    is_black_update = InputDimension("is_black_update")

    v_value_slot = InputDimension("v_value_slot")

    node_id       = InputDimension("node_id")
    is_fixed      = InputDimension("is_fixed")
    init_voltage  = InputDimension("init_voltage")
    iter_parity   = InputDimension("iter_parity")
    # NEW: per-update-token current-injection contribution (scaled int volts).
    i_inj_slot    = InputDimension("i_inj_slot")

    nbr_off   = [InputDimension(f"nbr_off_{k}")    for k in range(D_MAX)]
    nbr_w     = [InputDimension(f"nbr_w_{k}")       for k in range(D_MAX)]
    nbr_valid = [InputDimension(f"nbr_valid_{k}")   for k in range(D_MAX)]

    readout_offset = InputDimension("readout_offset")

    # ---- DSL: update-token compute --------------------------------------
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
    # NEW: include i_inj_slot in v_jacobi BEFORE the SOR relaxation.
    v_jacobi = persist(
        sum_wv * (1.0 / SCALE) + 1.0 * i_inj_slot,
        name="v_jacobi",
    )

    v_old_raw = fetch(
        value=v_value_slot,
        query=position + (-2 * N),
        key=position,
        clear_key=1 - is_v_emission,
        tie_break="latest",
    )
    v_old = persist(v_old_raw, name="v_old")

    v_sor_free = persist(
        (1.0 - omega) * v_old + omega * v_jacobi,
        name="v_sor_free",
    )

    v_new = persist(
        reglu(init_voltage, is_fixed) + reglu(v_sor_free, 1 - is_fixed),
        name="v_new",
    )

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

    for n in range(N):
        is_red = n in red_set
        prefix = "red" if is_red else "blk"
        fixed_flag = 1 if pc.is_fixed[n] else 0
        v0 = int(round(pc.fixed_voltage[n] * SCALE))
        i_inj_scaled = int(round(pc.i_inj_norm[n] * SCALE)) if not pc.is_fixed[n] else 0

        out = pc.out_edges_norm[n]
        nbr_pairs = list(out)[:D_MAX]

        per_slot_off: list[int] = []
        per_slot_w:   list[int] = []
        per_slot_valid: list[int] = []
        for k in range(D_MAX):
            if k < len(nbr_pairs):
                nbr_id, w = nbr_pairs[k]
                delta = 2 * (cidx[nbr_id] - cidx[n]) + 1
                if (not is_red) and (nbr_id in red_set):
                    off = delta
                else:
                    off = -2 * N + delta
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
                + i_inj_scaled * i_inj_slot
            )
            for k in range(D_MAX):
                if per_slot_valid[k]:
                    emb = emb + per_slot_off[k]   * nbr_off[k]
                    emb = emb + per_slot_w[k]     * nbr_w[k]
                    emb = emb + 1                 * nbr_valid[k]
            input_tokens[f"up_{prefix}_{n}_p{p}"] = emb

    ro_off = 2 * cidx[target_node] - 2 * N + 1
    input_tokens["readout"] = (
        1 * is_readout_tok + ro_off * readout_offset
    )

    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

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
        "has_i_sources": any(abs(x) > 1e-12 for x in pc.i_inj_norm),
    }
    return input_tokens, output_tokens, meta


class RBSORCircuitMachineI:
    """Drop-in replacement for RBSORCircuitMachine that supports I-sources."""

    def __init__(
        self,
        parsed: ParsedCircuitWithI,
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
        input_tokens, output_tokens, meta = build_rbsor_circuit_graph_isource(
            self.parsed, self.target_node, self.T, self.omega,
            self.red_order, self.black_order,
            v_step=self.v_step, k_levels=self.k_levels,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
