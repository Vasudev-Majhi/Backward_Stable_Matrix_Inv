"""Investigate DSL ↔ reference drift on CKT_0085 (DSL gives 4.20V, ref 2.55V).

Hypothesis: the reference uses integer floor div (`acc // SCALE`) while the
DSL uses float multiply (`sum_wv * (1.0 / SCALE)`). Each iteration's
sub-quantum difference gets amplified by ω≈1.93 over 200 iterations.

This script runs three solvers side-by-side:
  - INT  : current rbsor_reference (integer floor everywhere)
  - FLOAT: float arithmetic that emulates what the DSL actually computes
  - DSL  : already known from the transformer run (4.20V on CKT_0085)

If FLOAT matches the DSL's 4.20V, the float-vs-floor difference is confirmed
as the root cause.
"""
from __future__ import annotations

import _path  # noqa: F401

import json
import os
import sys

from coloring import two_color
from experiments.spectrum import compute_rho_kappa
from parse import parse_netlist
from rbsor_reference import (
    K_LEVELS,
    SCALE,
    V_STEP,
    auto_T_rbsor,
    omega_opt,
    rbsor_solve_parsed,
)


def _quantize(v_scaled: float, v_step: int = V_STEP, k_levels: int = K_LEVELS) -> int:
    """Same v_k argmax as the DSL: round-to-nearest, clip to [0, k_levels-1]."""
    k = int(round(v_scaled / v_step))
    if k < 0:
        k = 0
    elif k >= k_levels:
        k = k_levels - 1
    return k * v_step


def rbsor_float(
    pc, target, T, omega, red_order, black_order,
    v_step: int = V_STEP, k_levels: int = K_LEVELS,
    trace_iters: list[int] | None = None,
) -> tuple[float, list[tuple[int, list[int], list[int]]]]:
    """Float arithmetic mirroring DSL semantics:
       v_jacobi = acc / SCALE   (true float divide, not floor)
       sor      = (1-ω)*v_old + ω*v_jacobi   (float)
       emit     = quantize(sor)    (matches argmax of quadratic in DSL)

    Returns (v_target, trace) where trace is a list of (iter, v_int, v_emitted).
    """
    n = pc.num_nodes
    # Initial values: each free node starts at 0, fixed nodes at their source v.
    # All quantized so the t=0 reads see what init v_k tokens carry.
    v = [_quantize(int(round(pc.fixed_voltage[i] * SCALE)), v_step, k_levels)
         if pc.is_fixed[i]
         else _quantize(0.0, v_step, k_levels)
         for i in range(n)]

    red_set = set(red_order)
    trace = []
    trace_iters = set(trace_iters or [])

    def update(node: int, v_prev_node: int, v_for_neighbors: list[int]) -> int:
        if pc.is_fixed[node]:
            return _quantize(int(round(pc.fixed_voltage[node] * SCALE)),
                             v_step, k_levels)
        acc = sum(w * v_for_neighbors[nb] for nb, w in pc.out_edges_norm[node])
        v_jacobi = acc / SCALE                # float divide, NOT floor
        sor = (1.0 - omega) * v_prev_node + omega * v_jacobi   # float
        return _quantize(sor, v_step, k_levels)

    for t in range(T):
        v_prev = list(v)
        for node in red_order:
            v[node] = update(node, v_prev[node], v_prev)
        v_for_blk = [v[i] if i in red_set else v_prev[i] for i in range(n)]
        for node in black_order:
            v[node] = update(node, v_prev[node], v_for_blk)
        if t in trace_iters:
            trace.append((t, list(v), [vi // v_step for vi in v]))

    return v[target] / SCALE, trace


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CRAFT_DATASET",
        r"dataset/circuit_dataset_rv.jsonl",
    )
    cids = sys.argv[2:] if len(sys.argv) > 2 else ["CKT_0085", "CKT_0075", "CKT_0129", "CKT_0134"]

    with open(path) as f:
        circuits = [json.loads(l) for l in f]

    print(f"{'Circuit':<10} {'N':>3} {'rho':>8} {'omega':>6} {'T':>5}  "
          f"{'INT_pred':>9} {'FLOAT_pred':>11} {'truth':>8}  "
          f"{'INT_err':>8} {'FLOAT_err':>10}  conf")
    print("-" * 110)

    for cid in cids:
        c = next(x for x in circuits if x["ID"] == cid)
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = c["Ground_Truth_Vout"]
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        red, black, conflicts = two_color(pc)
        T = auto_T_rbsor(pc.num_nodes, omega)

        int_pred = rbsor_solve_parsed(pc, target, T, omega, red, black)
        float_pred, _ = rbsor_float(pc, target, T, omega, red, black)

        print(f"{cid:<10} {pc.num_nodes:>3} {rho:>8.4f} {omega:>6.3f} {T:>5}  "
              f"{int_pred:>9.4f} {float_pred:>11.4f} {truth:>8.4f}  "
              f"{abs(int_pred-truth):>8.4f} {abs(float_pred-truth):>10.4f}  "
              f"{len(conflicts)}")


if __name__ == "__main__":
    main()
