"""Pure-Python Chebyshev-Accelerated Discrete Jacobi (CADJ) reference solver.

Mirrors the DSL semantics exactly so per-iteration vectors can be diffed for
validation. Twin of `cadj_interpreter.py`.

The recurrence (one step at iteration k, for free node i):
    v_jacobi(v_k)[i] = sum_j w_ij * v_k[j]              # neighbor sum
    v_{k+1}[i]       = alpha_k * v_jacobi[i]
                     + beta_k  * v_k[i]
                     + gamma_k * v_{k-1}[i]

Coefficients (precomputed once per circuit from spectral bounds lam_min, lam_max
of D^{-1} A_ff):
    sigma = (lam_max + lam_min) / (lam_max - lam_min)
    rho_0 = 1/sigma
    rho_k = 1 / (2*sigma - rho_{k-1})       for k >= 1
    k = 0:   alpha_0 = 1/sigma,  beta_0 = 1 - 1/sigma,  gamma_0 = 0
    k >= 1:  alpha_k = 2*rho_k,  beta_k = 2*rho_k*(sigma - 1),  gamma_k = 1 - 2*sigma*rho_k

Identity: alpha_k + beta_k + gamma_k == 1 for all k. (Verified in __main__.)

Convergence rate: rho_cheby = (1 - sqrt(1 - rho_J^2)) / (1 + sqrt(1 - rho_J^2))
where rho_J = 1 - lam_min is the Jacobi spectral radius. Same asymptotic rate
as optimal RB-SOR but coefficients decay smoothly, eliminating quantum
overshoot in the discrete vocab.
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

import math

from parse import ParsedCircuit, parse_netlist

SCALE     = 10000     # voltage scaling — same as Jacobi/RB-SOR
V_STEP    = 500       # 0.05 V default (interpreter uses 500 / 10000 = 0.05)
K_LEVELS  = 480       # 0..23.95 V default


def chebyshev_coefficients(
    lam_min: float, lam_max: float, T: int,
) -> tuple[list[float], list[float], list[float]]:
    """Precompute (alphas, betas, gammas) of length T from spectral bounds.

    Degenerate cases (where Chebyshev acceleration buys nothing):
      - lam_max == lam_min (1-node system): plain Jacobi (alpha=1, beta=gamma=0)
      - kappa = lam_max/lam_min < 2 (well-conditioned): plain Jacobi
    """
    if T <= 0:
        return [], [], []
    if lam_max <= 0 or lam_min <= 0 or lam_max <= lam_min:
        # Degenerate / well-conditioned -> plain Jacobi.
        return [1.0] * T, [0.0] * T, [0.0] * T
    kappa = lam_max / lam_min
    if kappa < 2.0:
        return [1.0] * T, [0.0] * T, [0.0] * T
    sigma = (lam_max + lam_min) / (lam_max - lam_min)
    alphas: list[float] = []
    betas: list[float] = []
    gammas: list[float] = []

    # k = 0 special case (no v_{-1} dependency): alpha_0 = 1/sigma, beta_0 = 1 - 1/sigma, gamma_0 = 0
    alphas.append(1.0 / sigma)
    betas.append(1.0 - 1.0 / sigma)
    gammas.append(0.0)

    rho = 1.0 / sigma   # rho_0
    for k in range(1, T):
        rho = 1.0 / (2.0 * sigma - rho)
        alphas.append(2.0 * rho)
        betas.append(2.0 * rho * (sigma - 1.0))
        gammas.append(1.0 - 2.0 * sigma * rho)
    return alphas, betas, gammas


def auto_T_cadj(
    n_nodes: int, lam_min: float, lam_max: float, t_floor: int = 200,
) -> int:
    """Iteration count target for CADJ.

    rho_J = 1 - lam_min  (Jacobi spectral radius for D^{-1}A_ff with rows
    summed to ~1 in our normalization). rho_cheby = (1 - sqrt(1 - rho_J^2)) /
    (1 + sqrt(1 - rho_J^2)). Aim for ~4 decimal digits + 50 margin.
    """
    if lam_min <= 0 or lam_max <= 0:
        return t_floor
    rho_j = max(min(1.0 - lam_min, 0.9999999), 0.0)
    if rho_j <= 0:
        return t_floor
    s = math.sqrt(max(1.0 - rho_j * rho_j, 1e-12))
    contraction = max((1.0 - s) / (1.0 + s), 0.001)
    digits = 4.0
    return max(t_floor, int(math.ceil(digits / -math.log10(contraction))) + 50)


def _quantize_to_vk(
    v_scaled: float, v_step: int = V_STEP, k_levels: int = K_LEVELS,
) -> int:
    """Quantize a scaled voltage to the nearest v_k token value.

    Mirrors the DSL: argmax of `2*(k*v_step)*v_score - (k*v_step)^2` is
    equivalent to round-to-nearest of `v_score / v_step`. Clamps to vocab.
    """
    k = int(round(v_scaled / v_step))
    if k < 0:
        k = 0
    elif k >= k_levels:
        k = k_levels - 1
    return k * v_step


def cadj_solve_parsed(
    pc: ParsedCircuit,
    target: int,
    T: int,
    lam_min: float,
    lam_max: float,
    v_step: int = V_STEP,
    k_levels: int = K_LEVELS,
) -> float:
    """Run T Chebyshev-accelerated iterations. Returns voltage at `target` (V).

    Mirrors the DSL exactly: every emitted v_k value is quantized via
    `_quantize_to_vk`, fixed nodes always re-emit their source voltage, free
    nodes get clipped to [0, k_levels*v_step] (mirrors reglu(..., 1-is_fixed)).
    """
    n = pc.num_nodes
    init_scaled = [
        _quantize_to_vk(int(round(pc.fixed_voltage[i] * SCALE)), v_step, k_levels)
        if pc.is_fixed[i]
        else _quantize_to_vk(0, v_step, k_levels)
        for i in range(n)
    ]
    v: list[int] = list(init_scaled)
    v_old: list[int] = list(init_scaled)   # v_{k-1}; init = v_0 (so iter 1's v_old = v_0)

    alphas, betas, gammas = chebyshev_coefficients(lam_min, lam_max, T)

    for k in range(T):
        a, b, g = alphas[k], betas[k], gammas[k]
        v_new: list[int] = [0] * n
        for i in range(n):
            if pc.is_fixed[i]:
                v_new[i] = init_scaled[i]
                continue
            # Jacobi neighbor sum (in scaled units):
            acc = 0
            for nb, w in pc.out_edges_norm[i]:
                acc += w * v[nb]
            v_jac = acc / SCALE   # float divide, matches DSL `* (1.0/SCALE)`
            # Three-term Chebyshev recurrence:
            out = a * v_jac + b * v[i] + g * v_old[i]
            v_new[i] = _quantize_to_vk(out, v_step, k_levels)
        v_old = v
        v = v_new

    return v[target] / SCALE


def cadj_solve(
    netlist: str, target: int, T: int, lam_min: float, lam_max: float,
    v_step: int = V_STEP, k_levels: int = K_LEVELS,
) -> float:
    pc = parse_netlist(netlist)
    return cadj_solve_parsed(pc, target, T, lam_min, lam_max, v_step, k_levels)


# ── Built-in self-tests (Verification steps 1-3 from the plan) ─────────────

def _test_coefficient_identity(T: int = 200, n_cases: int = 20) -> None:
    """Step 1: assert alpha_k + beta_k + gamma_k == 1 for k=0..T-1."""
    import random
    random.seed(0)
    for _ in range(n_cases):
        lam_min = random.uniform(0.001, 1.0)
        lam_max = lam_min + random.uniform(0.5, 1.999 - lam_min)
        a, b, g = chebyshev_coefficients(lam_min, lam_max, T)
        for k in range(T):
            s = a[k] + b[k] + g[k]
            assert abs(s - 1.0) < 1e-9, f"sum != 1 at k={k}: {s}"
    print(f"  [step 1] alpha+beta+gamma == 1 verified for {n_cases} (lam_min, lam_max) pairs, T={T}")


def _test_vs_direct_solve(dataset_path: str, n_circuits: int = 10) -> None:
    """Step 2: cadj_reference vs experiments.spectrum.direct_solve on first n circuits."""
    import json

    from experiments.spectrum import compute_eigenvalue_bounds, direct_solve

    print(f"  [step 2] cadj_reference vs direct_solve, first {n_circuits} circuits:")
    n_pass = 0
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i >= n_circuits:
                break
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            target = int(c["Target_Node"])
            lam_min, lam_max = compute_eigenvalue_bounds(pc)
            if lam_max == float("inf"):
                print(f"    {c['ID']:<10} N={pc.num_nodes:<3} skipped (singular)")
                continue
            T = auto_T_cadj(pc.num_nodes, lam_min, lam_max)
            cadj_v = cadj_solve_parsed(pc, target, T, lam_min, lam_max)
            try:
                ref_v = direct_solve(pc, target)
            except Exception as e:
                print(f"    {c['ID']:<10} direct_solve failed: {e}")
                continue
            err = abs(cadj_v - ref_v)
            ok = err <= 0.05
            n_pass += int(ok)
            print(f"    {c['ID']:<10} N={pc.num_nodes:<3} T={T:<5} "
                  f"lam=[{lam_min:.4f},{lam_max:.4f}] "
                  f"cadj={cadj_v:.4f} direct={ref_v:.4f} err={err:.4f} {'OK' if ok else 'FAIL'}")
    print(f"    -> {n_pass}/{min(n_circuits, i+1)} match within 0.05V")


def _test_hard_tier(dataset_path: str) -> None:
    """Step 3: cadj_reference vs rbsor_reference on representative Hard circuits."""
    import json
    import sys as _sys

    _rbsor_dir = _path._PARENT + "/rbsor"
    if _rbsor_dir not in _sys.path:
        _sys.path.insert(0, _rbsor_dir)

    from coloring import two_color
    from experiments.spectrum import compute_eigenvalue_bounds, compute_rho_kappa, direct_solve
    from rbsor_reference import (
        auto_T_rbsor as _rb_auto_T, omega_opt as _rb_omega, rbsor_solve_parsed,
    )

    targets = ["CKT_0085", "CKT_0103", "CKT_0124", "CKT_0137", "CKT_0145"]
    by_id = {}
    with open(dataset_path) as f:
        for line in f:
            c = json.loads(line)
            by_id[c["ID"]] = c

    print(f"  [step 3] CADJ vs RB-SOR reference on Hard-tier sample:")
    for cid in targets:
        if cid not in by_id:
            print(f"    {cid:<10} (missing)")
            continue
        c = by_id[cid]
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = c["Ground_Truth_Vout"]
        try:
            direct = direct_solve(pc, target)
        except Exception:
            direct = float("nan")
        rho, _ = compute_rho_kappa(pc)
        omega = _rb_omega(rho)
        T_rb = _rb_auto_T(pc.num_nodes, omega)
        red, black, _ = two_color(pc)
        rb_v = rbsor_solve_parsed(pc, target, T_rb, omega, red, black)

        lam_min, lam_max = compute_eigenvalue_bounds(pc)
        T_cadj = auto_T_cadj(pc.num_nodes, lam_min, lam_max)
        cadj_v = cadj_solve_parsed(pc, target, T_cadj, lam_min, lam_max)

        rb_err = abs(rb_v - truth)
        cadj_err = abs(cadj_v - truth)
        better = "CADJ wins" if cadj_err < rb_err else "RB-SOR wins" if rb_err < cadj_err else "TIE"
        print(f"    {cid:<10} N={pc.num_nodes:<3} truth={truth:.4f} direct={direct:.4f}")
        print(f"               RB-SOR T={T_rb:<5} pred={rb_v:.4f} err={rb_err:.4f}")
        print(f"               CADJ   T={T_cadj:<5} lam=[{lam_min:.4f},{lam_max:.4f}] "
              f"pred={cadj_v:.4f} err={cadj_err:.4f}  [{better}]")


if __name__ == "__main__":
    import os
    import sys as _sys

    print("=" * 72)
    print("CADJ reference self-test")
    print("=" * 72)

    _test_coefficient_identity()

    dataset_path = _sys.argv[1] if len(_sys.argv) > 1 else os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    )

    if os.path.exists(dataset_path):
        _test_vs_direct_solve(dataset_path)
        _test_hard_tier(dataset_path)
    else:
        print(f"  [skip steps 2-3] dataset not found at {dataset_path}")
