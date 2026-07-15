"""Direct sensitivity-matrix solver for resistor networks.

For a resistor network with free nodes F and fixed (source) nodes P:
    A_FF * v_F + A_FP * v_P = 0
    => v_F = -A_FF^{-1} * A_FP * v_P  =  S * v_P

S[i,j] is the voltage sensitivity of free node i to source j.
For purely resistive networks: S[i,j] >= 0 and sum_j S[i,j] == 1.

Build-time: compute S = -solve(A_FF, A_FP) via float64 numpy.
This handles condition numbers up to ~10^10 correctly.
"""
from __future__ import annotations

import _path  # noqa: F401

import numpy as np

from parse import ParsedCircuit, parse_netlist

from cadj_reference import K_LEVELS, SCALE, V_STEP

M_SOURCES_MAX = 10   # max fixed (source) nodes; dataset has at most 5


def _quantize_to_vk(v_scaled: float, v_step: int = V_STEP, k_levels: int = K_LEVELS) -> int:
    k = int(round(v_scaled / v_step))
    if k < 0:
        k = 0
    elif k >= k_levels:
        k = k_levels - 1
    return k * v_step


def build_sensitivity_matrix(
    pc: ParsedCircuit,
) -> tuple[list[int], list[int], np.ndarray]:
    """Return (free_nodes, fixed_nodes, S) where S[i,j] = v_free_i / v_fixed_j.

    S has shape (len(free), len(fixed)).
    Raises numpy.linalg.LinAlgError if A_FF is singular.
    """
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g
        A[b, b] += g
        A[a, b] -= g
        A[b, a] -= g

    free  = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]

    if not free:
        return free, fixed, np.zeros((0, len(fixed)), dtype=np.float64)
    if not fixed:
        return free, fixed, np.zeros((len(free), 0), dtype=np.float64)

    fi = np.array(free)
    pi = np.array(fixed)
    A_ff = A[np.ix_(fi, fi)]
    A_fp = A[np.ix_(fi, pi)]
    # S = -A_FF^{-1} A_FP
    S = np.linalg.solve(A_ff, -A_fp)
    return free, fixed, S


def direct_solve_parsed(
    pc: ParsedCircuit,
    target: int,
    v_step: int = V_STEP,
    k_levels: int = K_LEVELS,
) -> float:
    """Return the quantized voltage at `target` (in Volts) via exact linear solve.

    Uses build_sensitivity_matrix and applies quantization identical to the DSL:
    argmax v_k formula rounds to nearest k, clamped to [0, k_levels-1].
    """
    if pc.is_fixed[target]:
        v_scaled = int(round(pc.fixed_voltage[target] * SCALE))
        return _quantize_to_vk(v_scaled, v_step, k_levels) / SCALE

    free, fixed, S = build_sensitivity_matrix(pc)
    free_map = {n: i for i, n in enumerate(free)}

    if target not in free_map:
        return 0.0

    ti = free_map[target]
    v_out_scaled = 0.0
    for j, fn in enumerate(fixed):
        v_out_scaled += S[ti, j] * pc.fixed_voltage[fn] * SCALE

    return _quantize_to_vk(v_out_scaled, v_step, k_levels) / SCALE
