"""E4, E5: spectral radius ρ(M) and condition number κ(A) per circuit.

For a resistor network with Kirchhoff nodal admittance matrix A (restricted
to free nodes), the Jacobi iteration matrix is M = D^{-1}(L + U) where
A = D - L - U. Here:
  - D = diagonal of A (self-conductances)
  - L + U = off-diagonal parts (neighbor conductances, negated)

We compute:
  - ρ(M) = max |eigenvalue(M)|
  - κ(A) = cond(A_free_free) via numpy

Uses RAW conductances (1/resistance), not the normalized int weights stored
in pc.out_edges_norm. Parallel resistors between the same pair of nodes are
summed (conductances add). GND (node 0) and V-source nodes are excluded
from the free-free submatrix.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from parse import ParsedCircuit, parse_netlist


def build_conductance_data(pc: ParsedCircuit) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return (free_node_ids, A_free_free, D_free).

    A is built from raw conductances g = 1/R with parallel-edge merging.
    A_free_free is the submatrix restricted to free (non-fixed) node rows/cols.
    D_free is the diagonal of A_free_free (self-admittance of each free node).
    """
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)

    # Merge parallel resistors: conductances add
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g
        A[b, b] += g
        A[a, b] -= g
        A[b, a] -= g

    free = [i for i in range(N) if not pc.is_fixed[i]]
    if not free:
        return [], np.zeros((0, 0)), np.zeros(0)

    idx = np.array(free)
    A_ff = A[np.ix_(idx, idx)]
    D_ff = np.diag(A_ff).copy()
    return free, A_ff, D_ff


def compute_rho_kappa(pc: ParsedCircuit) -> tuple[float, float]:
    """Return (spectral_radius of Jacobi iteration matrix M, condition number of A).

    Edge cases:
      - If there are no free nodes: returns (0.0, 1.0).
      - If a free node has zero self-admittance (disconnected): returns (float('inf'),
        float('inf')) so callers can flag it.
    """
    free, A_ff, D_ff = build_conductance_data(pc)
    if len(free) == 0:
        return 0.0, 1.0

    if np.any(D_ff <= 0):
        return float("inf"), float("inf")

    # M = D^{-1} (L + U) = I - D^{-1} A
    D_inv = np.diag(1.0 / D_ff)
    M = np.eye(len(free)) - D_inv @ A_ff

    try:
        eigs = np.linalg.eigvals(M)
        rho = float(np.max(np.abs(eigs)))
    except np.linalg.LinAlgError:
        rho = float("inf")

    try:
        kappa = float(np.linalg.cond(A_ff))
    except np.linalg.LinAlgError:
        kappa = float("inf")

    return rho, kappa


def compute_eigenvalue_bounds(pc: ParsedCircuit) -> tuple[float, float]:
    """Return (lam_min, lam_max) of the Jacobi-preconditioned matrix D^{-1} A_ff.

    Uses the identity: eigvals(D^{-1} A_ff) = 1 - eigvals(M) where M is the
    Jacobi iteration matrix already computed by `compute_rho_kappa`. CADJ uses
    these bounds to precompute Chebyshev coefficients offline.

    Edge cases:
      - No free nodes: returns (0.0, 2.0) (caller should handle separately)
      - Singular row (zero diagonal): returns (0.0, float('inf'))
    """
    free, A_ff, D_ff = build_conductance_data(pc)
    if len(free) == 0:
        return 0.0, 2.0
    if np.any(D_ff <= 0):
        return 0.0, float("inf")

    # eigvals(D^{-1} A_ff) for symmetric M-matrix ⟶ all real, in (0, 2].
    M = np.diag(1.0 / D_ff) @ A_ff
    try:
        eigs = np.linalg.eigvals(M).real
        lam_min = float(eigs.min())
        lam_max = float(eigs.max())
    except np.linalg.LinAlgError:
        return 0.0, float("inf")
    return lam_min, lam_max


def direct_solve(pc: ParsedCircuit, target: int) -> float:
    """Exact solution for the target node via np.linalg.solve (ground-truth sanity).

    This solves A_ff v_free = b where b[i] = Σ g_ij * V_fixed_j over fixed neighbors j.
    Returns the voltage at `target` (or the fixed voltage if target is fixed).
    """
    if pc.is_fixed[target]:
        return float(pc.fixed_voltage[target])

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
    idx = np.array(free)
    A_ff = A[np.ix_(idx, idx)]
    b = np.zeros(len(free))
    for fi, fn in enumerate(fixed):
        v = pc.fixed_voltage[fn]
        if v == 0.0:
            continue
        for li, ln in enumerate(free):
            b[li] += -A[ln, fn] * v  # moving A[ln,fn]*v to RHS
    v_free = np.linalg.solve(A_ff, b)
    free_map = {n: i for i, n in enumerate(free)}
    return float(v_free[free_map[target]])


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/circuit_dataset_rv.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            rho, kappa = compute_rho_kappa(pc)
            direct = direct_solve(pc, int(c["Target_Node"]))
            print(
                f"{c['ID']:<10} N={pc.num_nodes:<3} "
                f"rho={rho:.6f}  kappa={kappa:.2e}  "
                f"direct={direct:.4f}V  truth={c['Ground_Truth_Vout']:.4f}V"
            )
