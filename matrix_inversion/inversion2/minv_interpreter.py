"""Per-matrix Jacobi DSL graph for matrix inversion.

Algorithm: Jacobi iteration for A·x = b, where the matrix coefficients
A[i,k]/A[i,i] and 1/A[i,i] are baked into the transformer's weight
tensors as InputDimension static embeddings. One forward pass = one
Jacobi update step. The runner injects b[i] and x[i] at runtime.

Update rule:
    x_new[i] = (1/A[i,i]) * b[i]  +  sum_{k!=i} (-A[i,k]/A[i,i]) * x[k]

Signed coefficients are split into _pos/_neg halves (both >= 0) so that
reglu(value, weight) stays valid (reglu requires weight >= 0).

Token sequence (per inference, one column of I):
    pos 0         : start
    pos 1..n      : init_0..init_{n-1}
    pos n+1..n+Tn : (up_0..up_{n-1}) x T
    pos n+Tn+1    : halt

Design notes:
  - x_value_slot and b_value_slot are InputDimensions (not Persist) to
    avoid cycle detection in the MILP scheduler. Runner injects them.
  - D_MAX = n-1: every row depends on every other row (dense matrix).
  - No v_k vocabulary, no <PRED> token. Runner reads x_buf directly.
  - No SCALE: we want full float64 precision.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

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


def build_inversion_graph(A: np.ndarray, T: int):
    """Return (input_tokens, output_tokens, meta).

    A must be square, non-singular, and diagonally dominant.
    meta holds dim objects needed by build.py for slot extraction.
    """
    n = A.shape[0]
    D_MAX = n - 1  # each row has exactly n-1 neighbors

    one = _graph.one
    position = _graph.position

    # ---- Input dimensions ------------------------------------------------
    is_init_tok   = InputDimension("is_init_tok")
    is_update_tok = InputDimension("is_update_tok")

    # Runtime-injected at up_i tokens. Plain InputDimension (not Persist)
    # to avoid the cycle that would result from persisting x_new -> fetch -> x_value_slot.
    x_value_slot  = InputDimension("x_value_slot")
    b_value_slot  = InputDimension("b_value_slot")

    node_id = InputDimension("node_id")

    # Per-row baked coefficients: 1/A[i,i] split into pos/neg halves.
    inv_a_pos_dim = InputDimension("inv_a_pos")
    inv_a_neg_dim = InputDimension("inv_a_neg")

    # Per-neighbor-slot fields (D_MAX = n-1 neighbors per row).
    nbr_off   = [InputDimension(f"nbr_off_{k}")   for k in range(D_MAX)]
    nbr_w_pos = [InputDimension(f"nbr_w_pos_{k}") for k in range(D_MAX)]
    nbr_w_neg = [InputDimension(f"nbr_w_neg_{k}") for k in range(D_MAX)]
    nbr_valid = [InputDimension(f"nbr_valid_{k}") for k in range(D_MAX)]

    # ---- DSL: compute graph (per up_i, applied uniformly) ----------------

    # Mask: only init and update tokens carry valid x_value_slot.
    fetch_mask = 1 - is_init_tok - is_update_tok

    # Sum of neighbor contributions: sum_k (-A[i,k]/A[i,i]) * x[k]
    sum_term = Expression()
    for k in range(D_MAX):
        x_to_k = fetch(
            value=x_value_slot,
            query=position + nbr_off[k],
            key=position,
            clear_key=fetch_mask,
            tie_break="latest",
        )
        # Signed weight split: neg_a_norm[k] can be either sign.
        pos_contrib = reglu(x_to_k, nbr_w_pos[k])
        neg_contrib = reglu(x_to_k, nbr_w_neg[k])
        # Gate with nbr_valid to mask unused slots.
        pos_gated = reglu(pos_contrib, nbr_valid[k])
        neg_gated = reglu(neg_contrib, nbr_valid[k])
        sum_term = sum_term + pos_gated - neg_gated

    # b[i] / A[i,i] term, signed.
    b_pos = reglu(b_value_slot, inv_a_pos_dim)
    b_neg = reglu(b_value_slot, inv_a_neg_dim)
    # Without the reglu wrapper below, MILP uses reglu_21 (slot 3 at L2 FFN time,
    # which is 0 when x[neighbors]=0) as the gate for b_pos+, killing the b_term
    # at the first iteration. Wrapping through nbr_valid[0] (always=1 for n>=2)
    # forces the MILP to use slot 17 (nbr_valid_0) as the gate instead.
    if D_MAX > 0:
        b_pos = reglu(b_pos, nbr_valid[0])
        b_neg = reglu(b_neg, nbr_valid[0])
    b_term = b_pos - b_neg

    x_new = persist(b_term + sum_term, name="x_new")

    # ---- input_tokens ----------------------------------------------------
    input_tokens: dict[str, Expression] = {}

    input_tokens["start"] = Expression()
    input_tokens["halt"]  = Expression()

    for i in range(n):
        input_tokens[f"init_{i}"] = (
            1 * is_init_tok
            + i * node_id
        )

    for i in range(n):
        inv_aii = 1.0 / A[i, i]
        others = [k for k in range(n) if k != i]  # n-1 non-self neighbors

        emb = (
            1 * is_update_tok
            + i * node_id
            + max(0.0, +inv_aii) * inv_a_pos_dim
            + max(0.0, -inv_aii) * inv_a_neg_dim
        )

        for slot, k in enumerate(others):
            # Position of up_k at iter t-1 relative to up_i at iter t:
            # offset = (k - i) - n  (see §3.3 of plan)
            off_k = (k - i) - n
            neg_a_norm = -A[i, k] * inv_aii  # signed coefficient
            w_pos = max(0.0, +neg_a_norm)
            w_neg = max(0.0, -neg_a_norm)

            emb = emb + off_k * nbr_off[slot]
            emb = emb + w_pos * nbr_w_pos[slot]
            emb = emb + w_neg * nbr_w_neg[slot]
            emb = emb + 1 * nbr_valid[slot]

        # If D_MAX > n-1 (shouldn't happen here since D_MAX = n-1 exactly),
        # fill remaining with far-negative offset to push query off-map.
        for slot in range(len(others), D_MAX):
            very_negative = -10 * n - 1000
            emb = emb + very_negative * nbr_off[slot]

        input_tokens[f"up_{i}"] = emb

    # Every non-start token carries one=1 (iJacobi convention).
    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens ---------------------------------------------------
    # Reference x_new from up_i output tokens. This keeps x_new in slot_of:
    # without a consumer in output_tokens, the MILP computes x_new into a
    # temporary slot that is not tracked in slot_of and cannot be read by
    # the runner. Mirroring iJacobi's v_score_source pattern (Lesson 7).
    # We never call model.head(), so the actual output scores don't matter.
    output_tokens: dict[str, Expression] = {
        tok: Expression() for tok in input_tokens
    }
    for tok in output_tokens:
        if tok.startswith("up_"):
            output_tokens[tok] = x_new

    # Save meta BEFORE auto_name so dim objects are the originals present in slot_of.
    # auto_name may wrap Expressions in new dims (iJacobi Lesson 7), making the
    # post-auto_name objects invisible to slot_of lookups.
    meta = {
        "n": n,
        "T": T,
        "x_value_slot_dim": x_value_slot,
        "b_value_slot_dim": b_value_slot,
        "x_new_expr": x_new,
    }
    auto_name(locals())
    return input_tokens, output_tokens, meta


class MinvMachine:
    """Wraps build_inversion_graph for a given matrix A."""

    def __init__(self, A: np.ndarray, T: int):
        self.A = A
        self.T = T

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_inversion_graph(self.A, self.T)
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
