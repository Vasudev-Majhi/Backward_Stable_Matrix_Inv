"""SPICE netlist -> merged LU+direct token sequence.

Layout (length 2N + 3*n_free + 4):
  pos 0                                start
  pos 2j+1   (j=0..N-1)                skip
  pos 2j+2   (j=0..N-1)                v_init_j
  pos 2N+1+i           (i=0..n_f-1)    rhs_i
  pos 2N+1+n_f+i                       fwd_i
  pos 2N+1+2*n_f+i                     bck_{n_f-1-i}      (REVERSE)
  pos 2N+1+3*n_f                       readout
  pos 2N+1+3*n_f+1                     <PRED>
  pos 2N+1+3*n_f+2                     halt

The bck block is reversed: bck_{n_f-1} comes first, bck_0 last. This matches
inversion2's convention so back-sub fetches read x_j (j>i) from positions
strictly to the left.
"""
from __future__ import annotations

import _path  # noqa: F401

from parse import ParsedCircuit, parse_netlist  # type: ignore
from cadj_reference import K_LEVELS, SCALE, V_STEP  # type: ignore

PREDICTED = "<PRED>"


def _v_token_for(voltage_v: float, v_step: int, k_levels: int) -> str:
    v_scaled = voltage_v * SCALE
    k = int(round(v_scaled / v_step))
    k = max(0, min(k_levels - 1, k))
    return f"v_{k}"


def tokenize_lu_direct(
    netlist: str,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuit, dict]:
    """Return (tokens, pc, layout_meta).

    layout_meta exposes free/fixed indices and the per-stage position bases so
    the runner knows where to read hidden state and patch the V cache.
    """
    vs = V_STEP if v_step is None else int(v_step)
    kl = K_LEVELS if k_levels is None else int(k_levels)

    pc = parse_netlist(netlist)
    N = pc.num_nodes
    free = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]
    n_free = len(free)

    toks: list[str] = ["start"]
    for j in range(N):
        toks.append("skip")
        toks.append(_v_token_for(pc.fixed_voltage[j], vs, kl))

    # Stages 1..3: rhs, fwd, bck (bck reversed).
    pos_rhs_base = 1 + 2 * N            # = 2N+1
    pos_fwd_base = pos_rhs_base + n_free
    pos_bck_base = pos_fwd_base + n_free
    pos_readout = pos_bck_base + n_free

    for i in range(n_free):
        toks.append(f"rhs_{i}")
    for i in range(n_free):
        toks.append(f"fwd_{i}")
    for slot in range(n_free):
        i = n_free - 1 - slot
        toks.append(f"bck_{i}")

    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    layout = {
        "N": N,
        "n_free": n_free,
        "n_fixed": len(fixed),
        "free": free,
        "fixed": fixed,
        "pos_rhs_base": pos_rhs_base,        # rhs_i at pos_rhs_base + i
        "pos_fwd_base": pos_fwd_base,        # fwd_i at pos_fwd_base + i
        "pos_bck_base": pos_bck_base,        # bck_{n-1-slot} at pos_bck_base + slot
        "pos_readout": pos_readout,
    }
    return toks, pc, layout
