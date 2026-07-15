"""Numpy reference for Idea2: explicit A_FF⁻¹ path (validation only).

Uses -np.linalg.inv(A_FF) @ A_FP to compute S. Produces identical results
to direct_reference.build_sensitivity_matrix (which uses linalg.solve).
This file exists to validate that the explicit-inverse path agrees with the
solve path before we replace it with the LU transformer.
"""
from __future__ import annotations

import os
import sys

_HERE        = os.path.dirname(os.path.abspath(__file__))
_CADJ        = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
for _p in (_CLAUDE_FILES, _CADJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

import numpy as np

from cadj_reference import K_LEVELS, SCALE, V_STEP
from direct_reference import _quantize_to_vk
from parse import ParsedCircuit, parse_netlist


def build_sensitivity_matrix_inv(
    pc: ParsedCircuit,
) -> tuple[list[int], list[int], np.ndarray]:
    """Return (free_nodes, fixed_nodes, S) using S = -A_FF⁻¹ @ A_FP."""
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g; A[a, b] -= g; A[b, a] -= g

    free  = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]

    if not free:
        return free, fixed, np.zeros((0, len(fixed)), dtype=np.float64)
    if not fixed:
        return free, fixed, np.zeros((len(free), 0), dtype=np.float64)

    fi = np.array(free)
    pi = np.array(fixed)
    A_FF = A[np.ix_(fi, fi)]
    A_FP = A[np.ix_(fi, pi)]
    S = -np.linalg.inv(A_FF) @ A_FP
    return free, fixed, S


def idea2_solve_parsed(
    pc: ParsedCircuit,
    target: int,
    v_step: int = V_STEP,
    k_levels: int = K_LEVELS,
) -> float:
    """Return quantized voltage at target via explicit-inverse S."""
    if pc.is_fixed[target]:
        v_scaled = int(round(pc.fixed_voltage[target] * SCALE))
        return _quantize_to_vk(v_scaled, v_step, k_levels) / SCALE

    free, fixed, S = build_sensitivity_matrix_inv(pc)
    free_map = {n: i for i, n in enumerate(free)}
    if target not in free_map:
        return 0.0

    ti = free_map[target]
    v_out_scaled = sum(
        S[ti, j] * pc.fixed_voltage[fn] * SCALE
        for j, fn in enumerate(fixed)
    )
    return _quantize_to_vk(v_out_scaled, v_step, k_levels) / SCALE


def validate_vs_direct(dataset_path: str, n_circuits: int = 10, tol: float = 1e-9) -> None:
    """Verify idea2_reference agrees with direct_reference to tol on first n circuits."""
    import json
    from direct_reference import build_sensitivity_matrix

    passed = failed = 0
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i >= n_circuits:
                break
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            free_d, fixed_d, S_d = build_sensitivity_matrix(pc)
            free_i, fixed_i, S_i = build_sensitivity_matrix_inv(pc)
            if S_d.size == 0:
                continue
            err = float(np.max(np.abs(S_d - S_i)))
            ok = err < tol
            passed += int(ok); failed += int(not ok)
            print(f"  {c['ID']:<10} N={pc.num_nodes:<3} max_S_err={err:.2e} {'OK' if ok else 'FAIL'}")
    print(f"  -> {passed}/{passed+failed} agree within {tol:.0e}")


if __name__ == "__main__":
    import os as _os
    dp = _os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    )
    validate_vs_direct(dp, n_circuits=154)
