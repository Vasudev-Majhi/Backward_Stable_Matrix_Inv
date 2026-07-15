"""N4: FEM (5-point stencil) 2D Poisson on a unit square.

Generates the SPICE netlist via the existing
`benchmarks/pde/poisson_2d.py`, parses it (with I-source extension),
builds an A_FF, runs CRAFT inversion, and compares against
scipy.sparse.linalg.spsolve and the analytical sin(pi*x)*sin(pi*y) solution.

Output: results/N4_fem_poisson.csv
"""
from __future__ import annotations
import os, sys, csv, time, json
sys.path.insert(0, "matrix_inversion/inversion2")
sys.path.insert(0, "craft")
sys.path.insert(0, "craft/benchmarks")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _bootstrap  # noqa: F401
from build_lu import build_for_matrix
import runner_lu

# Use existing Poisson generator and the I-source parser
from pde.poisson_2d import poisson_grid_to_spice
from ext.parse_isource import parse_netlist_isource  # type: ignore


def fem_aff_apf(rows: int, cols: int, top_bc=0.0, bottom_bc=0.0,
                left_bc=0.0, right_bc=0.0, forcing=1.0):
    """Build A_FF, A_FP, target_node from a 2D Poisson netlist."""
    netlist, target_node, analytical = poisson_grid_to_spice(
        rows, cols, top_bc, bottom_bc, left_bc, right_bc, forcing)
    pc = parse_netlist_isource(netlist)
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for (a, b, ohms) in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g
        A[a, b] -= g; A[b, a] -= g
    free = [i for i in range(N) if not pc.is_fixed[i]]
    A_FF = A[np.ix_(free, free)]
    # Right-hand side for free nodes: -A_FP * v_fixed + current injections
    # For Dirichlet=0 BC and unit forcing, b = forcing * 1 (one ampere at each interior)
    b = np.full(len(free), forcing, dtype=np.float64)
    return A_FF, b, analytical, len(free), target_node


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "N4_fem_poisson.csv")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs("/tmp/N4_models", exist_ok=True)
    rows = []
    fields = ["mesh", "n_dof", "craft_build_s", "craft_infer_s",
              "max_abs_err_craft_vs_scipy", "max_abs_err_craft_vs_dense_inv",
              "status", "error"]
    for size in (4, 6, 8, 10, 12, 16):
        row = {f: "" for f in fields}
        row["mesh"] = f"{size}x{size}"
        try:
            A_FF, b, _ana, n, _ = fem_aff_apf(size, size, forcing=1.0)
            row["n_dof"] = n
            print(f"\n=== Mesh {size}x{size}  n_dof={n} ===", flush=True)
            t0 = time.time()
            info = build_for_matrix(A_FF, model_dir="/tmp/N4_models")
            row["craft_build_s"] = time.time() - t0
            t0 = time.time()
            X_craft = runner_lu.invert(A_FF, info["model_path"])
            row["craft_infer_s"] = time.time() - t0
            # u_craft = X_craft @ b
            u_craft = X_craft @ b
            # scipy reference
            A_sp = sp.csc_matrix(A_FF)
            u_scipy = spla.spsolve(A_sp, b)
            u_dense = np.linalg.inv(A_FF) @ b
            row["max_abs_err_craft_vs_scipy"] = float(np.max(np.abs(u_craft - u_scipy)))
            row["max_abs_err_craft_vs_dense_inv"] = float(np.max(np.abs(u_craft - u_dense)))
            row["status"] = "ok"
            print(f"  CRAFT vs scipy: {row['max_abs_err_craft_vs_scipy']:.2e}", flush=True)
            print(f"  CRAFT vs dense inv: {row['max_abs_err_craft_vs_dense_inv']:.2e}", flush=True)
        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)[:200]
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
        rows.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
