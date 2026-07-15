"""Per-circuit Direct Sensitivity DSL graph.

Algorithm: exact linear solve of A_FF * v_F = -A_FP * v_P baked into weights.
For a resistor network, v_F = S * v_P where S = -A_FF^{-1} A_FP.

Token sequence (length 2N+4):
  pos 0           : start
  pos 2j+1        : skip    (j = 0..N-1)
  pos 2j+2        : v_init_j
  pos 2N+1        : readout
  pos 2N+2        : <PRED>
  pos 2N+3        : halt

The readout token fetches each fixed (source) node's v_init_j from the
initial block via a fixed negative offset, multiplies by the precomputed
sensitivity coefficient S[target, j], and sums. The argmax output scoring
formula then picks the nearest v_k token.

v_init_j is at pos 2j+2; readout is at pos 2N+1.
Fetch offset: (2j+2) - (2N+1) = 2j - 2N + 1  (always <= -1 for j < N).

Sensitivity coefficients S[target,j] are non-negative for resistive networks
(maximum principle) — safe to use as reglu gates directly.

COEF_SCALE: integer precision for sensitivity coefficients stored in embedding.
10000 gives 4 decimal digits of accuracy for S values in [0,1].
"""
from __future__ import annotations

import _path  # noqa: F401

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

from cadj_reference import K_LEVELS, SCALE, V_STEP
from direct_reference import M_SOURCES_MAX, build_sensitivity_matrix

H_OUT = 1e5
COEF_SCALE = 10000


def build_direct_circuit_graph(
    pc: ParsedCircuit,
    target_node: int,
    v_step: int | None = None,
    k_levels: int | None = None,
    S_override=None,
):
    """Return (input_tokens, output_tokens, meta). Assumes reset_graph() called."""
    v_step = V_STEP if v_step is None else int(v_step)
    k_levels = K_LEVELS if k_levels is None else int(k_levels)

    one = _graph.one
    position = _graph.position
    N = pc.num_nodes

    # ---- Compute sensitivity matrix at build time -------------------------
    if S_override is not None:
        free  = [i for i in range(N) if not pc.is_fixed[i]]
        fixed = [i for i in range(N) if pc.is_fixed[i]]
        S = S_override
    else:
        free, fixed, S = build_sensitivity_matrix(pc)
    assert len(fixed) <= M_SOURCES_MAX, (
        f"Circuit has {len(fixed)} fixed nodes but M_SOURCES_MAX={M_SOURCES_MAX}"
    )

    if target_node in [fn for fn in fixed]:
        # Target is a fixed node — sensitivity is trivial (identity).
        # Build a degenerate graph: readout just fetches its own v_init.
        target_is_fixed = True
        target_fixed_v_scaled = int(round(
            pc.fixed_voltage[target_node] * SCALE
        ))
    else:
        target_is_fixed = False
        free_map = {n: i for i, n in enumerate(free)}
        ti = free_map[target_node]

    # ---- Input dimensions ------------------------------------------------
    is_v_emission  = InputDimension("is_v_emission")
    is_readout_tok = InputDimension("is_readout_tok")
    is_skip_tok    = InputDimension("is_skip_tok")
    v_value_slot   = InputDimension("v_value_slot")

    # Per-source-slot dimensions (M_SOURCES_MAX slots).
    src_off   = [InputDimension(f"src_off_{k}")   for k in range(M_SOURCES_MAX)]
    sens_coef = [InputDimension(f"sens_coef_{k}") for k in range(M_SOURCES_MAX)]
    is_src    = [InputDimension(f"is_src_{k}")    for k in range(M_SOURCES_MAX)]

    # ---- DSL: readout token accumulates sensitivity-weighted voltages -----
    # v_out = sum_j  S[target,j] * v_init_fixed_j
    # Each source's v_init_j lives at pos 2j+2 in the initial block.
    # Fetch offset from readout (pos 2N+1): 2j+2 - (2N+1) = 2j-2N+1.

    v_direct_acc = Expression()   # accumulator; Expression() == 0
    for k in range(M_SOURCES_MAX):
        v_src_k = fetch(
            value=v_value_slot,
            query=position + src_off[k],
            key=position,
            clear_key=1 - is_v_emission,
            tie_break="latest",
        )
        # reglu(v_src_k, is_src[k]) gates out unused slots (is_src_k == 0).
        # reglu(gated, sens_coef[k]) scales by S[target,j]*COEF_SCALE (integer >= 0).
        gated = reglu(v_src_k, is_src[k])
        weighted = reglu(gated, sens_coef[k])
        v_direct_acc = v_direct_acc + weighted

    # Divide by COEF_SCALE. The result is in the same scaled-integer units as
    # v_value_slot (k*v_step), so the output scoring formula works directly.
    v_out = persist(
        (1.0 / COEF_SCALE) * v_direct_acc,
        name="v_out",
    )

    v_score_source = persist(
        reglu(v_out, is_readout_tok),
        name="v_score_source",
    )
    emit_v_gate = persist(is_readout_tok, name="emit_v_gate")

    # ---- input_tokens ----------------------------------------------------
    input_tokens: dict[str, Expression] = {}
    input_tokens["start"] = Expression()
    input_tokens["skip"]  = 1 * is_skip_tok
    input_tokens["halt"]  = Expression()

    for k in range(k_levels):
        input_tokens[f"v_{k}"] = 1 * is_v_emission + (k * v_step) * v_value_slot

    # Readout: one token carrying all per-source offsets + sensitivity coefficients.
    # v_init_j is at pos 2j+2; readout is at pos 2N+1.
    # Fetch offset = (2j+2) - (2N+1) = 2*(j - N) + 1.
    readout_emb = 1 * is_readout_tok
    for slot in range(M_SOURCES_MAX):
        if slot < len(fixed):
            fn = fixed[slot]
            off = 2 * (fn - N) + 1   # always negative (fn < N)
            if target_is_fixed:
                # For fixed target: S[target, fn] = 1 if fn == target else 0.
                s_int = COEF_SCALE if fn == target_node else 0
            else:
                s_int = int(round(max(S[ti, slot], 0.0) * COEF_SCALE))
            readout_emb = readout_emb + off * src_off[slot]
            readout_emb = readout_emb + s_int * sens_coef[slot]
            readout_emb = readout_emb + 1 * is_src[slot]
        # else: slot unused — all zero (no term added)
    input_tokens["readout"] = readout_emb

    # Every non-start token carries one=1.
    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens: quadratic scoring --------------------------------
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
        "T": 0,
        "target_node": target_node,
        "n_fixed": len(fixed),
        "n_free": len(free),
        "v_out_expr": v_out,
    }
    return input_tokens, output_tokens, meta


class DirectCircuitMachine:
    def __init__(
        self,
        parsed: ParsedCircuit,
        target_node: int,
        v_step: int | None = None,
        k_levels: int | None = None,
        S_override=None,
    ):
        self.parsed = parsed
        self.target_node = target_node
        self.v_step = v_step
        self.k_levels = k_levels
        self.S_override = S_override

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_direct_circuit_graph(
            self.parsed, self.target_node,
            v_step=self.v_step, k_levels=self.k_levels,
            S_override=self.S_override,
        )
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
