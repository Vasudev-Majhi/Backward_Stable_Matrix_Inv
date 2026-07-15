"""Per-matrix LU-decomposition DSL graph for matrix inversion.

Algorithm: Doolittle LU factorization (A = L*U) is computed at build time.
L (unit lower-tri) and U (upper-tri) coefficients are baked into per-token
embeddings as InputDimension constants. Solving A*x = b becomes two
non-iterative triangular sweeps:

    Forward sub (Ly = b):
        y_0 = b_0   (since L[0,0] = 1)
        y_i = b_i - sum_{j<i} L[i,j] * y_j     for i = 1..n-1

    Back sub (Ux = y):
        x_{n-1} = y_{n-1} / U[n-1, n-1]
        x_i     = (y_i - sum_{j>i} U[i,j] * x_j) / U[i,i]   for i = n-2..0

Token sequence (per inference, one column of I):
    pos 0           : start
    pos 1..n        : init_0..init_{n-1}
    pos n+1..2n     : fwd_0..fwd_{n-1}                 (forward sub, IN ORDER)
    pos 2n+1..3n    : bck_{n-1}..bck_0                 (back sub, REVERSE)
    pos 3n+1        : halt

A single shared FFN computes both fwd_* and bck_* updates; the inactive
phase contributes 0 because its weight slots are 0 in that token's embedding.

Design notes:
  - x_value_slot, b_value_slot, y_input_slot are InputDimensions (not Persist)
    to avoid cycle detection in the MILP scheduler. Runner injects them.
  - The b-kicker (fires on fwd tokens) and y-kicker (fires on bck tokens)
    are gated by is_fwd_tok / is_bck_tok directly, NOT by lwd_valid_0 /
    uwd_valid_0 (which are 0 for fwd_0 and bck_{n-1}).
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

from lu_factor import doolittle


def build_inversion_graph_lu(A: np.ndarray):
    """Return (input_tokens, output_tokens, meta) for LU-decomposition transformer.

    No T parameter -- LU is non-iterative.
    A must be square and admit Doolittle LU (no zero pivots without pivoting).
    """
    n = A.shape[0]
    D_MAX_FWD = n - 1   # max neighbors in forward sub (worst case = fwd_{n-1})
    D_MAX_BCK = n - 1   # max neighbors in back sub (worst case = bck_0)

    # ---- LU factorization (build-time) ----------------------------------
    L, U = doolittle(A)
    if not np.allclose(L @ U, A, atol=1e-10):
        raise RuntimeError("LU factorization failed verification (L*U != A)")

    one = _graph.one
    position = _graph.position

    # ---- Input dimensions ------------------------------------------------
    is_init_tok = InputDimension("is_init_tok")
    is_fwd_tok  = InputDimension("is_fwd_tok")
    is_bck_tok  = InputDimension("is_bck_tok")

    # Runtime-injected slots (NOT Persist -- would cause MILP cycle).
    b_value_slot = InputDimension("b_value_slot")     # injected on fwd_i
    y_input_slot = InputDimension("y_input_slot")     # injected on bck_i
    x_value_slot = InputDimension("x_value_slot")     # attention value source (shared)

    node_id = InputDimension("node_id")

    # Per-row baked diagonals (sign-split for reglu compatibility).
    inv_l_pos_dim = InputDimension("inv_l_pos")       # +1/L[i,i]  (on fwd_i)
    inv_l_neg_dim = InputDimension("inv_l_neg")       # -1/L[i,i]  (on fwd_i)
    inv_u_pos_dim = InputDimension("inv_u_pos")       # +1/U[i,i]  (on bck_i)
    inv_u_neg_dim = InputDimension("inv_u_neg")       # -1/U[i,i]  (on bck_i)

    # Forward-sub neighbor bank (D_MAX_FWD slots).
    lwd_off   = [InputDimension(f"lwd_off_{k}")   for k in range(D_MAX_FWD)]
    lwd_w_pos = [InputDimension(f"lwd_w_pos_{k}") for k in range(D_MAX_FWD)]
    lwd_w_neg = [InputDimension(f"lwd_w_neg_{k}") for k in range(D_MAX_FWD)]
    lwd_valid = [InputDimension(f"lwd_valid_{k}") for k in range(D_MAX_FWD)]

    # Back-sub neighbor bank (D_MAX_BCK slots).
    uwd_off   = [InputDimension(f"uwd_off_{k}")   for k in range(D_MAX_BCK)]
    uwd_w_pos = [InputDimension(f"uwd_w_pos_{k}") for k in range(D_MAX_BCK)]
    uwd_w_neg = [InputDimension(f"uwd_w_neg_{k}") for k in range(D_MAX_BCK)]
    uwd_valid = [InputDimension(f"uwd_valid_{k}") for k in range(D_MAX_BCK)]

    # ---- DSL: compute graph ---------------------------------------------
    # Fetch mask: only init/fwd/bck tokens carry a valid x_value_slot.
    fetch_mask = 1 - is_init_tok - is_fwd_tok - is_bck_tok

    # Forward-sub sum: sum over j<i of (-L[i,j]/L[i,i]) * y_j.
    # (Since L is unit lower-tri, L[i,i]=1, so coef = -L[i,j].)
    sum_term_fwd = Expression()
    for k in range(D_MAX_FWD):
        fetch_lwd_k = fetch(
            value=x_value_slot,
            query=position + lwd_off[k],
            key=position,
            clear_key=fetch_mask,
            tie_break="latest",
        )
        pos_contrib = reglu(fetch_lwd_k, lwd_w_pos[k])
        neg_contrib = reglu(fetch_lwd_k, lwd_w_neg[k])
        pos_gated = reglu(pos_contrib, lwd_valid[k])
        neg_gated = reglu(neg_contrib, lwd_valid[k])
        sum_term_fwd = sum_term_fwd + pos_gated - neg_gated

    # Back-sub sum: sum over j>i of (-U[i,j]/U[i,i]) * x_j.
    sum_term_bck = Expression()
    for k in range(D_MAX_BCK):
        fetch_uwd_k = fetch(
            value=x_value_slot,
            query=position + uwd_off[k],
            key=position,
            clear_key=fetch_mask,
            tie_break="latest",
        )
        pos_contrib = reglu(fetch_uwd_k, uwd_w_pos[k])
        neg_contrib = reglu(fetch_uwd_k, uwd_w_neg[k])
        pos_gated = reglu(pos_contrib, uwd_valid[k])
        neg_gated = reglu(neg_contrib, uwd_valid[k])
        sum_term_bck = sum_term_bck + pos_gated - neg_gated

    # b-kicker: only fires on fwd tokens. Gate by is_fwd_tok directly --
    # NOT by lwd_valid_0, which is 0 on fwd_0 (no j<0 neighbors).
    b_pos = reglu(b_value_slot, inv_l_pos_dim)
    b_neg = reglu(b_value_slot, inv_l_neg_dim)
    b_pos = reglu(b_pos, is_fwd_tok)
    b_neg = reglu(b_neg, is_fwd_tok)
    b_term = b_pos - b_neg

    # y-kicker: only fires on bck tokens. Gate by is_bck_tok.
    y_pos = reglu(y_input_slot, inv_u_pos_dim)
    y_neg = reglu(y_input_slot, inv_u_neg_dim)
    y_pos = reglu(y_pos, is_bck_tok)
    y_neg = reglu(y_neg, is_bck_tok)
    y_term = y_pos - y_neg

    # Single persisted output. Inactive phase contributes 0 by construction.
    x_new = persist(b_term + sum_term_fwd + y_term + sum_term_bck, name="x_new")

    # ---- input_tokens ----------------------------------------------------
    input_tokens: dict[str, Expression] = {}

    input_tokens["start"] = Expression()
    input_tokens["halt"]  = Expression()

    for i in range(n):
        input_tokens[f"init_{i}"] = (
            1 * is_init_tok
            + i * node_id
        )

    # fwd_i: forward sub for row i, sequence position n+1+i.
    for i in range(n):
        inv_lii = 1.0 / L[i, i]   # = 1 always for Doolittle
        others = list(range(i))    # j in 0..i-1

        emb = (
            1 * is_fwd_tok
            + i * node_id
            + max(0.0, +inv_lii) * inv_l_pos_dim
            + max(0.0, -inv_lii) * inv_l_neg_dim
        )

        for slot, j in enumerate(others):
            # fwd_j is at pos n+1+j, fwd_i at pos n+1+i.  Diff = j-i (negative).
            off_kj = j - i
            coef = -L[i, j] * inv_lii
            w_pos = max(0.0, +coef)
            w_neg = max(0.0, -coef)
            emb = emb + off_kj * lwd_off[slot]
            emb = emb + w_pos  * lwd_w_pos[slot]
            emb = emb + w_neg  * lwd_w_neg[slot]
            emb = emb + 1      * lwd_valid[slot]

        # Pad unused fwd slots with very-negative offset to push attention off-map.
        for slot in range(len(others), D_MAX_FWD):
            very_negative = -10 * n - 1000
            emb = emb + very_negative * lwd_off[slot]

        input_tokens[f"fwd_{i}"] = emb

    # bck_i: back sub for row i, sequenced in REVERSE order.
    # bck_{n-1} at pos 2n+1, bck_{n-2} at pos 2n+2, ..., bck_0 at pos 3n.
    # General: bck_i at pos 2n+1 + (n-1-i) = 3n - i.
    for i in range(n):
        inv_uii = 1.0 / U[i, i]
        others = list(range(i + 1, n))   # j in i+1..n-1

        emb = (
            1 * is_bck_tok
            + i * node_id
            + max(0.0, +inv_uii) * inv_u_pos_dim
            + max(0.0, -inv_uii) * inv_u_neg_dim
        )

        for slot, j in enumerate(others):
            # bck_j at pos 3n-j, bck_i at pos 3n-i.  Diff = (3n-j)-(3n-i) = i-j.
            # Since j>i this is negative -> attention queries the past. Correct.
            off_ij = i - j
            coef = -U[i, j] * inv_uii
            w_pos = max(0.0, +coef)
            w_neg = max(0.0, -coef)
            emb = emb + off_ij * uwd_off[slot]
            emb = emb + w_pos  * uwd_w_pos[slot]
            emb = emb + w_neg  * uwd_w_neg[slot]
            emb = emb + 1      * uwd_valid[slot]

        for slot in range(len(others), D_MAX_BCK):
            very_negative = -10 * n - 1000
            emb = emb + very_negative * uwd_off[slot]

        input_tokens[f"bck_{i}"] = emb

    # Every non-start token carries one=1 (iJacobi convention).
    for tok in input_tokens:
        if tok != "start":
            input_tokens[tok][one] = 1

    # ---- output_tokens ---------------------------------------------------
    # Reference x_new from fwd_i / bck_i output tokens so MILP keeps it in slot_of.
    output_tokens: dict[str, Expression] = {
        tok: Expression() for tok in input_tokens
    }
    for tok in output_tokens:
        if tok.startswith("fwd_") or tok.startswith("bck_"):
            output_tokens[tok] = x_new

    # Save meta BEFORE auto_name so dim refs are the originals present in slot_of.
    meta = {
        "n": n,
        "x_value_slot_dim": x_value_slot,
        "b_value_slot_dim": b_value_slot,
        "y_input_slot_dim": y_input_slot,
        "x_new_expr": x_new,
        "L": L,
        "U": U,
    }
    auto_name(locals())
    return input_tokens, output_tokens, meta


class MinvMachineLU:
    """Wraps build_inversion_graph_lu for a given matrix A."""

    def __init__(self, A: np.ndarray):
        self.A = A

    def build(self) -> tuple[ProgramGraph, dict]:
        reset_graph()
        input_tokens, output_tokens, meta = build_inversion_graph_lu(self.A)
        pg = ProgramGraph(input_tokens, output_tokens)
        return pg, meta
