"""N4 diagnostic: log U[i,i] diagonal from scalar Doolittle for each FEM mesh.

Explains the four anomalous mesh sizes (4x4, 6x6, 12x12, 16x16) by showing
that the FEM stiffness matrix admits structurally near-zero pivots at these
sizes, where pivotless Doolittle is undefined. The two working sizes
(8x8, 10x10) have min |U[i,i]| above the 1e-12 cutoff.

Output: results/N4/fem_pivots.csv
        results/N4/fem_pivots.log
"""
from __future__ import annotations
import os, sys, csv, json

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "craft_release", "matrix_inversion", "inversion2"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft", "benchmarks"))

import numpy as np

from pde.poisson_2d import poisson_grid_to_spice
from ext.parse_isource import parse_netlist_isource  # type: ignore


def fem_aff(rows, cols, forcing=1.0):
    netlist, target_node, analytical = poisson_grid_to_spice(
        rows, cols, 0.0, 0.0, 0.0, 0.0, forcing)
    pc = parse_netlist_isource(netlist)
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for (a, b, ohms) in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g
        A[a, b] -= g; A[b, a] -= g
    free = [i for i in range(N) if not pc.is_fixed[i]]
    return A[np.ix_(free, free)]


def doolittle_diag(A, tol=1e-12):
    """Doolittle LU without pivoting — returns U-diagonal and detection of the
    first near-zero pivot. Mirrors the lu_pipeline/lu_factor.py behavior but
    instead of raising, returns full diagnostic info."""
    A = A.astype(np.float64).copy()
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros_like(A)
    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - L[k, :k] @ U[:k, j]
        if abs(U[k, k]) < tol:
            return {
                "n": n, "fail_at": k, "min_abs_u_diag": float(abs(U[k, k])),
                "u_diag": [float(U[i, i]) for i in range(k + 1)] + [float("nan")] * (n - k - 1),
                "cond": float(np.linalg.cond(A)),
                "status": "near_zero_pivot",
            }
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]
    u_diag = [float(U[i, i]) for i in range(n)]
    return {
        "n": n, "fail_at": -1,
        "min_abs_u_diag": float(min(abs(U[i, i]) for i in range(n))),
        "u_diag": u_diag,
        "cond": float(np.linalg.cond(A)),
        "status": "ok",
    }


def main():
    out_dir = os.path.join(HERE, "..", "results", "N4")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "fem_pivots.csv")
    out_log = os.path.join(out_dir, "fem_pivots.log")

    rows = []
    log_lines = []
    for size in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        try:
            A = fem_aff(size, size)
            d = doolittle_diag(A)
            d["mesh"] = f"{size}x{size}"
            rows.append({
                "mesh": d["mesh"], "n_dof": d["n"],
                "cond": d["cond"],
                "min_abs_u_diag": d["min_abs_u_diag"],
                "fail_at": d["fail_at"],
                "status": d["status"],
            })
            log_lines.append(
                f"{d['mesh']}  n={d['n']}  cond={d['cond']:.3e}  "
                f"min|U[i,i]|={d['min_abs_u_diag']:.3e}  status={d['status']}  "
                f"fail_at={d['fail_at']}"
            )
            # Save full U-diagonal as JSON sidecar.
            with open(os.path.join(out_dir, f"udiag_{size}x{size}.json"), "w") as f:
                json.dump(d, f, indent=2)
            print(log_lines[-1], flush=True)
        except Exception as e:
            rows.append({
                "mesh": f"{size}x{size}", "n_dof": -1, "cond": -1,
                "min_abs_u_diag": -1, "fail_at": -1, "status": f"error: {e!r}"[:200],
            })
            log_lines.append(f"{size}x{size}  ERROR: {e!r}")
            print(log_lines[-1], flush=True)

    fields = ["mesh", "n_dof", "cond", "min_abs_u_diag", "fail_at", "status"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    with open(out_log, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_log}")


if __name__ == "__main__":
    main()
