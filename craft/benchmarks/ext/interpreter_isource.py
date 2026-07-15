"""Per-circuit Jacobi DSL graph WITH current-source (I) support.

Mirrors `interpreter.py` exactly except for two additions:
  1. New InputDimension `i_inj_slot` carrying scaled-int volts per update token
     (= I_inj_norm[n] * SCALE).
  2. v_new_free now equals  sum_wv * (1/SCALE) + i_inj_slot.

When all i_inj_norm[n] == 0, the produced graph is mathematically identical
to the original interpreter.py output (the i_inj_slot dimension just stays at 0
in every embedding row). The d_model grows by 1 input-dim slot.

The standard `interpreter.py` is NOT modified; this file is consumed only by
benchmarks/ orchestrators that work with `ParsedCircuitWithI`.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

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
from interpreter import D_MAX, H_OUT, K_LEVELS, SCALE, V_STEP


def build_circuit_graph_isource(
    pc: ParsedCircuitWithI,
    target_node: int,
    T: int,
    v_step: int | None = None,
    k_levels: int | None = None,
):
    """Return (input_tokens, output_tokens, meta). Assumes reset_graph() was called."""
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes

    # ---- Input dimensions -----------------------------------------------
    is_update_tok   = InputDimension("is_update_tok")
    is_v_emission   = InputDimension("is_v_emission")
    is_readout_tok  = InputDimension("is_readout_tok")
    is_skip_tok     = InputDimension("is_skip_tok")

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

    # NEW: add i_inj_slot (already in scaled volts) to the Jacobi update.
    # sum_wv * (1/SCALE) is in scaled volts (V*SCALE); i_inj_slot is also in
    # scaled volts. Both addable directly.
    v_new_free = persist(
        sum_wv * (1.0 / SCALE) + 1.0 * i_inj_slot,
        name="v_new_free",
    )

    v_new = persist(
        reglu(init_voltage, is_fixed) + reglu(v_new_free, 1 - is_fixed),
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

    for n in range(N):
        fixed_flag = 1 if pc.is_fixed[n] else 0
        v0 = int(round(pc.fixed_voltage[n] * SCALE))
        # I_inj_norm[n] is in volts; bake it into the embedding as scaled volts.
        i_inj_scaled = int(round(pc.i_inj_norm[n] * SCALE)) if not pc.is_fixed[n] else 0

        out = pc.out_edges_norm[n]
        nbr_ids = [nb for (nb, _) in out][:D_MAX]
        nbr_ws  = [w  for (_, w)  in out][:D_MAX]

        for p in (0, 1):
            emb = (
                1 * is_update_tok
                + n * node_id
                + fixed_flag * is_fixed
                + v0 * init_voltage
                + p * iter_parity
                + i_inj_scaled * i_inj_slot
            )
            for k in range(D_MAX):
                if k < len(nbr_ids):
                    nbr_k = nbr_ids[k]
                    off_k = -2 * N + 2 * (nbr_k - n) + 1
                    emb = emb + off_k * nbr_off[k]
                    emb = emb + nbr_ws[k] * nbr_w[k]
                    emb = emb + 1 * nbr_valid[k]
            input_tokens[f"up_{n}_p{p}"] = emb

    ro_off = 2 * target_node - 2 * N + 1
    input_tokens["readout"] = (
        1 * is_readout_tok + ro_off * readout_offset
    )

    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens --------------------------------------------------
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
        "target_node": target_node,
        "target_v_expr": target_v_persist,
        "v_new_expr": v_new,
        "has_i_sources": any(abs(x) > 1e-12 for x in pc.i_inj_norm),
    }
    return input_tokens, output_tokens, meta


class CircuitMachineI:
    """Drop-in replacement for interpreter.CircuitMachine that supports I-sources.

    If pc has no I-sources (all i_inj_norm == 0), the produced graph is
    behaviorally identical to the standard CircuitMachine but has one extra
    (zero-coefficient) input dimension. That is acceptable: the regression
    check should compare voltages, not d_model.
    """
    def __init__(
        self,
        parsed: ParsedCircuitWithI,
        target_node: int,
        T: int,
        v_step: int | None = None,
        k_levels: int | None = None,
    ):
        self.parsed = parsed
        self.target_node = target_node
        self.v_step = v_step
        self.k_levels = k_levels
        self.T = T

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_circuit_graph_isource(
            self.parsed, self.target_node, self.T,
            v_step=self.v_step, k_levels=self.k_levels,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
