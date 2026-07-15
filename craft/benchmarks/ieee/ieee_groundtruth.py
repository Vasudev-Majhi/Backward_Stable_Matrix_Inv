"""Numpy reference solver for IEEE DC power-flow.

Solves B·θ = P directly via numpy linear-algebra (LU) to provide ground truth
for the compiled-transformer benchmark. The result is converted to the same
shifted/scaled "voltage" representation used by `ieee_to_netlist.dc_pf_to_netlist`
so it can be compared apples-to-apples with the model's prediction.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import numpy as np


def dc_pf_solve(
    case: dict,
    V_bias: float = 12.0,
    K: float = 30.0,
    target_bus: int | None = None,
) -> tuple[float, dict]:
    """Solve B·θ = P, return (V_target, all_voltages_dict).

    V_target = θ_target * K + V_bias  (the value the compiled transformer
    should predict). all_voltages_dict maps every bus id to its shifted/scaled
    voltage so callers can verify per-bus accuracy if desired.
    """
    base_mva = case["base_mva"]
    slack_bus = case["slack_bus"]
    buses = case["buses"]
    branches = case["branches"]

    bus_ids = sorted(b[0] for b in buses)
    bus_idx = {b: i for i, b in enumerate(bus_ids)}
    N = len(bus_ids)
    slack_idx = bus_idx[slack_bus]

    # Build B matrix
    B = np.zeros((N, N), dtype=np.float64)
    for (fbus, tbus, _r, x, _b) in branches:
        if x <= 0:
            continue
        i = bus_idx[fbus]
        j = bus_idx[tbus]
        b_ij = 1.0 / x
        B[i, i] += b_ij
        B[j, j] += b_ij
        B[i, j] -= b_ij
        B[j, i] -= b_ij

    # P vector (per unit)
    P = np.zeros(N)
    for (b, _typ, Pd, _Qd, Pg, _Qg) in buses:
        i = bus_idx[b]
        P[i] = (Pg - Pd) / base_mva

    # Strike out the slack row/col, solve B_red θ_red = P_red
    keep = [i for i in range(N) if i != slack_idx]
    B_red = B[np.ix_(keep, keep)]
    P_red = P[keep]

    theta_red = np.linalg.solve(B_red, P_red)

    theta_full = np.zeros(N)
    for kk, ii in enumerate(keep):
        theta_full[ii] = theta_red[kk]
    # theta_full[slack_idx] = 0 already

    # Convert to "voltage" representation
    voltages = {bus_ids[i]: theta_full[i] * K + V_bias for i in range(N)}

    if target_bus is None:
        # Default: largest-non-slack bus
        target_bus = max(b for b in bus_ids if b != slack_bus)

    return voltages[target_bus], voltages
