"""SPICE netlist -> Direct solver token sequence.

Compact layout — no iteration block:

  pos 0           : start
  pos 2*j+1       : skip          (for j in 0..N-1)
  pos 2*j+2       : v_init_j      (initial voltage of node j, quantized)
  pos 2N+1        : readout       (carries sensitivity coefficients)
  pos 2N+2        : <PRED>
  pos 2N+3        : halt

Total sequence length: 2N + 4 tokens (plus 1 PRED slot).
Readout is the only active-compute token; it fetches each source node's
v_init and accumulates sensitivity-weighted sum.
"""
from __future__ import annotations

from parse import ParsedCircuit, parse_netlist
from cadj_reference import K_LEVELS, SCALE, V_STEP

PREDICTED = "<PRED>"


def _v_token_for(voltage_v: float, v_step: int, k_levels: int) -> str:
    v_scaled = voltage_v * SCALE
    k = int(round(v_scaled / v_step))
    k = max(0, min(k_levels - 1, k))
    return f"v_{k}"


def tokenize_direct(
    netlist: str,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuit]:
    """Return (tokens, pc).

    The token sequence has exactly one PREDICTED slot — after the readout
    token — for the final voltage prediction.
    """
    vs = V_STEP if v_step is None else int(v_step)
    kl = K_LEVELS if k_levels is None else int(k_levels)

    pc = parse_netlist(netlist)
    N = pc.num_nodes

    toks: list[str] = ["start"]

    # Initial block: skip + v_init for each node.
    for j in range(N):
        toks.append("skip")
        toks.append(_v_token_for(pc.fixed_voltage[j], vs, kl))

    # Readout + prediction + halt.
    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    return toks, pc
