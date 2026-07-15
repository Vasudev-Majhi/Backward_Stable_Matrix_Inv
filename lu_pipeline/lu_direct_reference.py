"""Reference solver: A_FF * v_F = -A_FP * v_P via Doolittle LU + triangular solves.

This is the build-time-honest counterpart to direct_reference.py: it never
calls numpy.linalg.solve. It only does sparse-A accumulation and Doolittle LU,
then forward/back substitution by hand. The merged transformer reproduces
forward+back substitution token-by-token; this module is the regression
oracle used by Step 4 of the plan.
"""
from __future__ import annotations

import _path  # noqa: F401

import numpy as np

from parse import ParsedCircuit, parse_netlist  # type: ignore
from cadj_reference import K_LEVELS, SCALE, V_STEP  # type: ignore

from lu_factor import back_sub, doolittle, forward_sub

M_SOURCES_MAX = 10  # mirror direct_reference.M_SOURCES_MAX


def _quantize_to_vk(v_scaled: float, v_step: int = V_STEP, k_levels: int = K_LEVELS) -> int:
    k = int(round(v_scaled / v_step))
    if k < 0:
        k = 0
    elif k >= k_levels:
        k = k_levels - 1
    return k * v_step


def build_partition(pc: ParsedCircuit) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Return (free, fixed, A_FF, A_FP) without any linalg.solve call.

    A is the conductance matrix (sparse-accumulated), then partitioned into
    A_FF (|free| x |free|) and A_FP (|free| x |fixed|).
    """
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g
        A[b, b] += g
        A[a, b] -= g
        A[b, a] -= g

    free = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]
    if not free:
        return free, fixed, np.zeros((0, 0)), np.zeros((0, len(fixed)))
    if not fixed:
        return free, fixed, A[np.ix_(free, free)], np.zeros((len(free), 0))
    fi = np.array(free)
    pi = np.array(fixed)
    return free, fixed, A[np.ix_(fi, fi)], A[np.ix_(fi, pi)]


def lu_direct_solve_parsed(
    pc: ParsedCircuit,
    target: int,
    v_step: int = V_STEP,
    k_levels: int = K_LEVELS,
) -> float:
    """Voltage at `target`, computed via LU + forward/back sub.

    Mirrors direct_solve_parsed in every quantization detail; the only change
    is the linear solve (LU instead of np.linalg.solve).
    """
    if pc.is_fixed[target]:
        v_scaled = int(round(pc.fixed_voltage[target] * SCALE))
        return _quantize_to_vk(v_scaled, v_step, k_levels) / SCALE

    free, fixed, A_FF, A_FP = build_partition(pc)
    free_map = {n: i for i, n in enumerate(free)}
    if target not in free_map:
        return 0.0

    v_P = np.array([pc.fixed_voltage[fn] for fn in fixed], dtype=np.float64)
    b = -A_FP @ v_P
    L, U = doolittle(A_FF)
    y = forward_sub(L, b)
    x = back_sub(U, y)

    ti = free_map[target]
    v_out_scaled = float(x[ti]) * SCALE
    return _quantize_to_vk(v_out_scaled, v_step, k_levels) / SCALE


def lu_direct_solve(netlist: str, target: int,
                    v_step: int = V_STEP, k_levels: int = K_LEVELS) -> float:
    return lu_direct_solve_parsed(parse_netlist(netlist), target, v_step, k_levels)
