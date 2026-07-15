"""Convert an IEEE bus DC power-flow problem to a SPICE-style netlist.

DC power flow: B · θ = P,  where
  - B is the susceptance matrix (B_ii = Σ b_ik, B_ij = -b_ij = -1/x_ij)
  - θ are the bus voltage angles (slack bus θ_slack = 0)
  - P is the net injection per bus (Pg - Pd) in per-unit (S_base = 100 MVA)

Mapping to a netlist (V + R + I):
  - Slack bus  →  node 1  with a V-source `V1 1 0 V_bias` (V_bias > 0).
                  This shifts the entire system upward so the resulting
                  "voltages" (= θ_i + V_bias) are non-negative — required
                  by the K-level quantized vocab (0..24V at V_STEP=0.05V).
  - Each branch (i,j) of reactance x_ij  →  resistor R = x_ij between the
                  two corresponding nodes (the conductance 1/R = 1/x is the
                  branch susceptance, which is exactly the off-diagonal entry).
  - Each non-slack bus i with net injection P_i  →  current source
                  `I<n> i 0 (P_i * K)` where K is the angle-scaling factor.

The recovered angle for bus b:  θ_b = (V_b - V_bias) / K
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401


def dc_pf_to_netlist(
    case: dict,
    V_bias: float = 12.0,
    K: float = 30.0,
    target_bus: int | None = None,
    description: str | None = None,
) -> tuple[str, int, dict]:
    """Build a netlist for the DC power-flow problem of `case`.

    Returns (netlist_text, target_node_0idx_in_netlist, meta).

    `target_bus` defaults to the bus farthest from slack in topology
    (largest index that's connected). Caller can override.
    """
    name = case["name"]
    slack_bus = case["slack_bus"]
    base_mva = case["base_mva"]
    buses = case["buses"]
    branches = case["branches"]

    # Build a contiguous 1..N node mapping. Slack → node 1; all others
    # in original ascending bus order. Node 0 = SPICE GND (always).
    bus_ids = [b[0] for b in buses]
    bus_ids_sorted = sorted(bus_ids)
    # Slack first, then others
    others = [b for b in bus_ids_sorted if b != slack_bus]
    ordered = [slack_bus] + others
    bus_to_node = {b: (i + 1) for i, b in enumerate(ordered)}  # SPICE 1-indexed
    node_to_bus = {n: b for b, n in bus_to_node.items()}

    if target_bus is None:
        # Pick the largest non-slack bus id as the target (often a load tail).
        target_bus = max(others)
    target_node = bus_to_node[target_bus]

    lines: list[str] = [
        f"* DC PF {name} V_bias={V_bias} K={K} target_bus={target_bus}",
    ]
    if description:
        lines.append(f"* {description}")

    # V-source at slack bus → V_bias volts.
    slack_node = bus_to_node[slack_bus]
    lines.append(f"V1 {slack_node} 0 {V_bias:.4f}")

    # Branches → resistors with R = x_ij.
    r_count = 0
    for (fbus, tbus, _r, x, _b) in branches:
        if x <= 0:
            continue  # skip degenerate branches
        n_from = bus_to_node[fbus]
        n_to = bus_to_node[tbus]
        r_count += 1
        lines.append(f"R{r_count} {n_from} {n_to} {x:.6f}")

    # Net injections P_i = (Pg - Pd) / base_mva, then scale by K.
    # Slack bus injection is implicit (slack absorbs); we don't add I-source there.
    i_count = 0
    p_inj_pu: dict[int, float] = {}
    for (b, _typ, Pd, _Qd, Pg, _Qg) in buses:
        if b == slack_bus:
            continue
        p_pu = (Pg - Pd) / base_mva
        p_inj_pu[b] = p_pu
        if p_pu == 0.0:
            continue
        i_count += 1
        scaled_amps = p_pu * K
        n = bus_to_node[b]
        lines.append(f"I{i_count} {n} 0 {scaled_amps:.6f}")

    lines.append(".op")
    lines.append(".end")

    meta = {
        "case_name": name,
        "V_bias": V_bias,
        "K": K,
        "slack_bus": slack_bus,
        "target_bus": target_bus,
        "target_node": target_node,
        "bus_to_node": bus_to_node,
        "node_to_bus": node_to_bus,
        "p_inj_pu": p_inj_pu,
        "n_buses": len(buses),
        "n_branches": r_count,
        "n_isources": i_count,
    }

    return "\n".join(lines), target_node, meta
