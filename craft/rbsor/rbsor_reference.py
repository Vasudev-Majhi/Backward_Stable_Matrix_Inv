"""Pure-Python Forward Red-Black SOR reference solver.

Mirrors the integer arithmetic of the DSL exactly so per-iteration vectors
can be diffed for validation. Designed as the ground-truth twin of the
RB-SOR DSL graph in `rbsor_interpreter.py`.

Forward RB-SOR (one iteration):
    1. Red sweep: each red node updates using v_prev for ALL its neighbors
       (matches DSL Cases A and B in the offset table — both fetch from the
       previous iteration's emission).
    2. Black sweep: each black node updates using the JUST-UPDATED red
       values for red neighbors (DSL Case C, current-iter fetch) and
       v_prev for any black neighbors (conflict edges, DSL Case A).

SOR combination per node:
    v_new = (1 - omega) * v_old + omega * v_jacobi
where v_jacobi = (Σ w_k * v_neighbor) / SCALE.
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

import math

from parse import ParsedCircuit, parse_netlist

SCALE = 10000
V_STEP = 500          # quantization step in scaled units (matches interpreter.py)
K_LEVELS = 480        # v_k vocabulary size (0..23.95 V)


def omega_opt(rho_jacobi: float, floor: float | None = None) -> float:
    """Young's optimum for SOR: ω* = 2 / (1 + sqrt(1 - ρ_J²)).

    Clamped to [1.0, 1.99]. Returns 1.0 (Gauss-Seidel) when ρ_J >= 1
    (degenerate / non-converging Jacobi) or when ρ_J <= 0.

    Optional `floor` raises ω to at least the given value (use 1.5 if you
    want extra integer-truncation amplification on easy circuits at the
    cost of slower per-iter convergence).
    """
    if not (0.0 < rho_jacobi < 1.0) or math.isnan(rho_jacobi):
        omega = 1.0
    else:
        omega = 2.0 / (1.0 + math.sqrt(1.0 - rho_jacobi * rho_jacobi))
    omega = max(1.0, min(1.99, omega))
    if floor is not None:
        omega = max(omega, float(floor))
        omega = min(omega, 1.99)
    return omega


def auto_T_rbsor(n_nodes: int, omega: float, t_floor: int = 200) -> int:
    """Iteration count target for RB-SOR.

    Uses ρ_RBSOR ≈ |omega - 1| as the per-iter contraction; aims for ~4
    decimal digits below quantization plus a safety margin of 50 iterations.
    Caller may set t_floor (e.g. 500 for the v01 protocol).
    """
    contraction = max(abs(omega - 1.0), 0.01)
    digits = 4.0
    return max(t_floor, int(math.ceil(digits / -math.log10(contraction))) + 50)


def detect_divergence(
    pc, target: int, omega: float, red_order: list[int], black_order: list[int],
    iter_a: int = 50, iter_b: int = 100,
    v_step: int = V_STEP, k_levels: int = K_LEVELS,
) -> tuple[bool, float, float]:
    """Run Python RB-SOR up to iter_a and iter_b; return (diverging, err_a, err_b).

    Diverging = err_b > err_a * 1.05 (5% margin to dampen quantum noise).
    Used by orchestrators to decide RB-SOR vs Jacobi fallback per circuit.
    """
    pred_a = rbsor_solve_parsed(pc, target, iter_a, omega, red_order, black_order,
                                v_step=v_step, k_levels=k_levels)
    pred_b = rbsor_solve_parsed(pc, target, iter_b, omega, red_order, black_order,
                                v_step=v_step, k_levels=k_levels)
    # Use prediction-vs-prediction delta as a proxy for divergence — if the
    # solution is still moving > 1 quantum step between t=50 and t=100, it
    # has not settled (could be diverging or just converging slowly).
    # Combine with: did |prediction| explode past plausible voltage range?
    delta = abs(pred_b - pred_a)
    plausible_max = max(abs(pred_a), abs(pred_b))
    diverging = (plausible_max > 24.0) or (delta > 1.0 and pred_b != 0.0)
    return diverging, pred_a, pred_b


def _quantize_to_vk(v_scaled: float, v_step: int = V_STEP, k_levels: int = K_LEVELS) -> int:
    """Quantize a scaled voltage to the nearest v_k token value.

    Mirrors the DSL: every v_old fetched in iteration t is one of the discrete
    v_k embeddings from the previous iteration's emission, not an arbitrary
    float. The DSL's argmax of `2*(k*v_step)*v_score - (k*v_step)^2` is
    equivalent to round-to-nearest of `v_score / v_step`, so we use the same
    here (Python's int(round(...)) for round-half-to-even, matches numpy).
    """
    k = int(round(v_scaled / v_step))
    if k < 0:
        k = 0
    elif k >= k_levels:
        k = k_levels - 1
    return k * v_step


def rbsor_solve_parsed(
    pc: ParsedCircuit,
    target: int,
    T: int,
    omega: float,
    red_order: list[int],
    black_order: list[int],
    v_step: int = V_STEP,
    k_levels: int = K_LEVELS,
) -> float:
    """Run T forward RB-SOR sweeps. Returns voltage at `target` in volts.

    Mirrors the DSL exactly: each v written into the working vector is the
    quantized v_k that the model would emit (via _quantize_to_vk), so
    subsequent iterations read the same discrete value the transformer would.
    """
    n = pc.num_nodes
    # Initial values: fixed nodes carry their source voltage; free nodes start at 0.
    # Both get quantized so the t=0 reads see what the init v_k tokens carry.
    v = [_quantize_to_vk(int(round(pc.fixed_voltage[i] * SCALE)), v_step, k_levels)
         if pc.is_fixed[i]
         else _quantize_to_vk(0, v_step, k_levels)
         for i in range(n)]

    red_set = set(red_order)

    def update(node: int, v_prev_node: int, v_for_neighbors: list[int]) -> int:
        if pc.is_fixed[node]:
            # Fixed nodes always re-emit their source voltage (also quantized).
            return _quantize_to_vk(
                int(round(pc.fixed_voltage[node] * SCALE)), v_step, k_levels
            )
        # DSL semantics: sum_wv is a float-valued persisted dim (line 124-127
        # of interpreter.py); v_jacobi = sum_wv * (1.0 / SCALE) is a true float
        # divide, NOT an integer floor. Mirror that here — using `// SCALE`
        # under-estimates by up to 1 scaled unit per iteration, which ω near 2
        # amplifies into a full quantum-step drift over ~30 iters (validated
        # by debug_drift.py: int-floor ref disagreed with DSL by 1.65V on
        # CKT_0085, but float-divide reference matches the DSL exactly).
        acc = 0
        for nb, w in pc.out_edges_norm[node]:
            acc += w * v_for_neighbors[nb]
        v_jacobi = acc / SCALE
        sor = (1.0 - omega) * v_prev_node + omega * v_jacobi
        # reglu(v_sor_free, 1 - is_fixed) clips negatives to 0 for free nodes;
        # argmax over v_k vocab clips to [0, k_levels-1].
        return _quantize_to_vk(sor, v_step, k_levels)

    for _ in range(T):
        v_prev = list(v)
        for node in red_order:
            v[node] = update(node, v_prev[node], v_prev)
        v_for_blk = [v[i] if i in red_set else v_prev[i] for i in range(n)]
        for node in black_order:
            v[node] = update(node, v_prev[node], v_for_blk)

    return v[target] / SCALE


def rbsor_solve(
    netlist: str,
    target: int,
    T: int,
    omega: float,
    red_order: list[int],
    black_order: list[int],
) -> float:
    pc = parse_netlist(netlist)
    return rbsor_solve_parsed(pc, target, T, omega, red_order, black_order)


if __name__ == "__main__":
    import json
    import os
    import sys

    from coloring import two_color
    from experiments.spectrum import compute_rho_kappa
    from jacobi_reference import _auto_T, jacobi_solve_parsed

    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    )
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    tol = 0.05

    j_pass = r_pass = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            target = int(c["Target_Node"])
            truth = c["Ground_Truth_Vout"]

            T_j = _auto_T(pc.num_nodes)
            j_v = jacobi_solve_parsed(pc, target, T_j)

            rho, _ = compute_rho_kappa(pc)
            omega = omega_opt(rho)
            T_r = auto_T_rbsor(pc.num_nodes, omega)
            red, black, conflicts = two_color(pc)
            r_v = rbsor_solve_parsed(pc, target, T_r, omega, red, black)

            j_ok = abs(j_v - truth) <= tol
            r_ok = abs(r_v - truth) <= tol
            j_pass += int(j_ok)
            r_pass += int(r_ok)

            print(f"{c['ID']:<10} N={pc.num_nodes:<3} rho={rho:.4f} omega={omega:.3f} "
                  f"T_j={T_j:<5} T_r={T_r:<5} "
                  f"jac={j_v:.4f}{'P' if j_ok else 'F'} "
                  f"rb={r_v:.4f}{'P' if r_ok else 'F'} "
                  f"truth={truth:.4f} conf={len(conflicts)}")

    print(f"\nJacobi: {j_pass}/{limit}  RB-SOR: {r_pass}/{limit}  (tol={tol}V)")
