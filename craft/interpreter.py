"""Per-circuit Jacobi DSL graph.

Algorithm: plain Jacobi iteration. Per-circuit model (each circuit compiles
to its own ProgramGraph / model.bin).

Architecture:
  - K-level quantized output-token emission for v_new (avoids multi-byte
    decomposition; avoids the dim-DAG cycle that residual-feedback creates).
  - Update tokens attend to the previous iteration's emitted v-tokens at
    known relative positions (tokenizer bakes offsets per-neighbor-slot).
  - First "iteration" is the init layer: the tokenizer seeds v-tokens
    encoding each node's initial voltage (V for fixed, 0 for free).

Token sequence:
    start                                   pos 0
    (skip, init_v(0)), (skip, init_v(1)), ..., (skip, init_v(N-1))   pos 1..2N
    iter 0: (update(0,p=0), v_pred), (update(1,p=0), v_pred), ...    pos 2N+1..4N
    iter t: (update(n,p=t%2), v_pred) ...
    readout(target), final_v_pred           pos 2(T+1)N+1, 2(T+1)N+2
    halt                                    pos 2(T+1)N+3

Per-iteration token count: 2N.
Per-circuit vocab: start + skip + halt + readout + N*2 updates + K v-tokens.

Vocabulary-scale note:
  K=480 covers 0..24V at 0.05V step (v_value = k * 500 in scaled-by-10000
  integer units). Every v_k has embedding (is_v_emission=1, v_value_slot=k*500).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 — sys.path shim
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

from parse import ParsedCircuit

SCALE = 10000                # voltages represented as int(V * SCALE)
V_STEP = 500                 # quantization step in scaled units = 0.05 V
K_LEVELS = 480               # v-token vocab: 0..23.95V in 480 steps
D_MAX = 8                    # max neighbor slots per update token
H_OUT = 1e5                  # hardmax amplifier for output scoring


def build_circuit_graph(
    pc: ParsedCircuit,
    target_node: int,
    T: int,
    v_step: int | None = None,
    k_levels: int | None = None,
):
    """Return (input_tokens, output_tokens, meta). Assumes reset_graph() was called.

    Optional quantization overrides:
      v_step:   scaled-integer step between adjacent v_k tokens (default V_STEP=500 = 0.05V)
      k_levels: number of v_k tokens, covering 0 .. (k_levels-1)*v_step / SCALE volts
    """
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes

    # ---- Input dimensions -----------------------------------------------
    # Kind flags
    is_update_tok   = InputDimension("is_update_tok")
    is_v_emission   = InputDimension("is_v_emission")
    is_readout_tok  = InputDimension("is_readout_tok")
    is_skip_tok     = InputDimension("is_skip_tok")

    # Carried by v-emission tokens only: the scalar v value (pre-quantized).
    v_value_slot = InputDimension("v_value_slot")

    # Update-token embedded data
    node_id       = InputDimension("node_id")
    is_fixed      = InputDimension("is_fixed")
    init_voltage  = InputDimension("init_voltage")
    iter_parity   = InputDimension("iter_parity")

    # Per-neighbor-slot fields on update tokens (D_MAX slots).
    nbr_off   = [InputDimension(f"nbr_off_{k}")    for k in range(D_MAX)]
    nbr_w     = [InputDimension(f"nbr_w_{k}")       for k in range(D_MAX)]
    nbr_valid = [InputDimension(f"nbr_valid_{k}")   for k in range(D_MAX)]

    # Readout token: position-delta to the target node's final v-emission
    readout_offset = InputDimension("readout_offset")

    # ---- DSL: update-token compute --------------------------------------
    # For each slot k, fetch v at (position + nbr_off_k), masked to v-emissions.
    # Gate the whole fetch by nbr_valid_k so unused slots contribute 0.
    # Also gate by is_update_tok so the operation has no side-effects on
    # non-update tokens (strictly speaking, position+nbr_off fetches still
    # run everywhere, but their results are multiplied by nbr_valid which
    # is 0 on non-update tokens).

    v_tos: list[Expression] = []
    msgs: list[Expression]  = []
    for k in range(D_MAX):
        v_to_k = fetch(
            value=v_value_slot,
            query=position + nbr_off[k],
            key=position,
            clear_key=1 - is_v_emission,
            tie_break="latest",
        )
        # msg_k = w_k * v_to_k (as a ReGLU×ReGLU multiply). Non-negative
        # inputs: w_k is a pre-normalized conductance ≥ 0, v_to_k is voltage
        # ≥ 0 → reglu suffices.
        msg_k = reglu(v_to_k, nbr_w[k])   # w_k * v_to_k  (since w_k ≥ 0)
        # Gate by nbr_valid_k (0 or 1).
        msg_k_gated = reglu(msg_k, nbr_valid[k])
        v_tos.append(v_to_k)
        msgs.append(msg_k_gated)

    sum_wv = Expression()
    for m in msgs:
        sum_wv = sum_wv + m
    sum_wv = persist(sum_wv, name="sum_wv")

    # Weights pre-normalized: Σ w_k = SCALE (for valid slots), so
    # v_new_free = sum_wv / SCALE.
    v_new_free = persist(sum_wv * (1.0 / SCALE), name="v_new_free")

    # Fixed-node gate: v_new = is_fixed ? init_voltage : v_new_free.
    # Both reglu arguments are non-negative.
    v_new = persist(
        reglu(init_voltage, is_fixed) + reglu(v_new_free, 1 - is_fixed),
        name="v_new",
    )

    # ---- DSL: readout-token compute -------------------------------------
    # Fetch the target node's final v-emission via position offset.
    target_v = fetch(
        value=v_value_slot,
        query=position + readout_offset,
        key=position,
        clear_key=1 - is_v_emission,
        tie_break="latest",
    )
    target_v_persist = persist(target_v, name="target_v")

    # Output-token score source: use v_new on update tokens, target_v on readout.
    # Gate via kind flags (reglu).
    v_score_source = persist(
        reglu(v_new, is_update_tok) + reglu(target_v_persist, is_readout_tok),
        name="v_score_source",
    )
    emit_v_gate = persist(is_update_tok + is_readout_tok, name="emit_v_gate")

    # ---- input_tokens ---------------------------------------------------
    input_tokens: dict[str, Expression] = {}

    # start: zero embedding (convention).
    input_tokens["start"] = Expression()

    # skip: used between init-v tokens to maintain stride-2 layout.
    input_tokens["skip"] = 1 * is_skip_tok

    # v_k tokens: K-level quantized voltage emissions.
    for k in range(k_levels):
        input_tokens[f"v_{k}"] = 1 * is_v_emission + (k * v_step) * v_value_slot

    # halt (marks end; no compute).
    input_tokens["halt"] = Expression()

    # Update tokens: one per (node, parity).
    for n in range(N):
        fixed_flag = 1 if pc.is_fixed[n] else 0
        v0 = int(round(pc.fixed_voltage[n] * SCALE))

        # Collect outgoing edges for node n (merged-parallel weights).
        out = pc.out_edges_norm[n]

        # Per-neighbor-slot values. Unused slots -> 0s.
        nbr_ids = [nb for (nb, _) in out][:D_MAX]
        nbr_ws  = [w  for (_, w)  in out][:D_MAX]

        for p in (0, 1):
            emb = (
                1 * is_update_tok
                + n * node_id
                + fixed_flag * is_fixed
                + v0 * init_voltage
                + p * iter_parity
            )
            for k in range(D_MAX):
                if k < len(nbr_ids):
                    nbr_k = nbr_ids[k]
                    off_k = -2 * N + 2 * (nbr_k - n) + 1
                    emb = emb + off_k * nbr_off[k]
                    emb = emb + nbr_ws[k] * nbr_w[k]
                    emb = emb + 1 * nbr_valid[k]
                # else: slot unset → all zero
            input_tokens[f"up_{n}_p{p}"] = emb

    # Readout: position offset to the target's final v-emission.
    # Readout at pos 2TN + 2N + 1.
    # Target's last v at pos 2TN + 2*target + 2 (for target ∈ [0, N)).
    ro_off = 2 * target_node - 2 * N + 1
    input_tokens["readout"] = (
        1 * is_readout_tok + ro_off * readout_offset
    )

    # Every non-start token carries one=1.
    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens: quadratic scoring for v-token emission ----------
    # Every token in the vocab needs a scoring expression.
    output_tokens: dict[str, Expression] = {
        tok: Expression() for tok in input_tokens
    }

    # v_k score = H * emit_v_gate + 2*(k*v_step)*v_score_source - (k*v_step)²
    for k in range(k_levels):
        vk = k * v_step
        output_tokens[f"v_{k}"] = (
            H_OUT * emit_v_gate
            + (2 * vk) * v_score_source
            + Expression({one: -(vk * vk)})
        )

    # Halt after readout's v emission — model score 'halt' after the final v.
    # (The runner stops after emit_vout anyway; this is here so MILP gives
    # 'halt' an output slot.)
    output_tokens["halt"] = Expression({one: 0})

    auto_name(locals())
    meta = {
        "N": N,
        "T": T,
        "target_node": target_node,
        "target_v_expr": target_v_persist,
        "v_new_expr": v_new,
    }
    return input_tokens, output_tokens, meta


class CircuitMachine:
    def __init__(
        self,
        parsed: ParsedCircuit,
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
        input_tokens, output_tokens, meta = build_circuit_graph(
            self.parsed, self.target_node, self.T,
            v_step=self.v_step, k_levels=self.k_levels,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
