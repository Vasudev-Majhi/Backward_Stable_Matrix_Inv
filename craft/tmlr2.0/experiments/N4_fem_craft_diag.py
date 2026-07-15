"""N4 deeper diagnostic: CRAFT inversion error vs scalar Doolittle vs scipy
for each FEM mesh size, plus comparison against the parsed conductances and
the I-source rhs.

Hypothesis: the original N4_fem_poisson.py used `u_craft = X_craft @ b` with
`b = full(forcing)` (Neumann-style uniform forcing) which is NOT the actual
Poisson RHS encoded by the netlist's I-current sources. The netlist's I-sources
already inject -forcing into each interior node's KCL; building b separately
double-counts (or sign-flips) the forcing.

This script checks:
  1. Scalar Doolittle vs scipy on the same A_FF (sanity)
  2. CRAFT vs scalar Doolittle on the same A_FF
  3. The actual netlist b-vector from current sources
"""
from __future__ import annotations
import os, sys, csv, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
# inversion2 LAST so its _bootstrap.py is found first (it adds the right
# transformer-vm path). Order matters: craft/_bootstrap.py points to
# ~/craft_release/transformer-vm which doesn't exist on all hosts.
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft", "benchmarks"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "matrix_inversion", "inversion2"))

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")
import numpy as np
import _bootstrap  # noqa: F401
from build_lu import build_for_matrix
import runner_lu
from pde.poisson_2d import poisson_grid_to_spice
from ext.parse_isource import parse_netlist_isource  # type: ignore


def build_aff_and_b(rows, cols, forcing=1.0):
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
    A_FF = A[np.ix_(free, free)]
    # Build RHS from I-sources if the parser exposes them; fall back to uniform.
    b_from_isrc = np.zeros(len(free), dtype=np.float64)
    isources = getattr(pc, "isources", None) or getattr(pc, "current_sources", None)
    if isources:
        free_idx = {n: k for k, n in enumerate(free)}
        for src in isources:
            # Common format: (node_from, node_to, amperes) — sign convention
            # depends on parser; use absolute then dot with -1 if needed.
            n_from, n_to, amps = src[0], src[1], src[2]
            if n_from in free_idx:
                b_from_isrc[free_idx[n_from]] -= amps  # current LEAVING node
            if n_to in free_idx:
                b_from_isrc[free_idx[n_to]] += amps    # current ENTERING node
    b_uniform = np.full(len(free), forcing, dtype=np.float64)
    return A_FF, b_from_isrc, b_uniform, len(free), target_node


def main():
    out_dir = os.path.join(HERE, "..", "results", "N4")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("/tmp/N4_diag_models", exist_ok=True)

    fields = ["mesh", "n_dof", "isrc_nnz",
              "scalar_doolittle_err_vs_scipy",
              "craft_err_vs_scipy",
              "craft_err_vs_doolittle",
              "max_inv_diff_craft_vs_dense",
              "build_s", "infer_s"]
    rows = []

    for size in (4, 6, 8, 10, 12):
        row = {f: "" for f in fields}
        row["mesh"] = f"{size}x{size}"
        try:
            A_FF, b_isrc, b_uniform, n, _ = build_aff_and_b(size, size)
            row["n_dof"] = n
            row["isrc_nnz"] = int((b_isrc != 0).sum())
            print(f"\n=== {size}x{size}  n={n}  isrc_nnz={row['isrc_nnz']} ===", flush=True)

            # scipy reference
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            u_scipy = spla.spsolve(sp.csc_matrix(A_FF), b_uniform)

            # scalar Doolittle (numpy.linalg.solve as scalar reference)
            u_dool = np.linalg.solve(A_FF, b_uniform)
            row["scalar_doolittle_err_vs_scipy"] = float(np.max(np.abs(u_dool - u_scipy)))

            # CRAFT
            t0 = time.time()
            info = build_for_matrix(A_FF, model_dir="/tmp/N4_diag_models")
            row["build_s"] = time.time() - t0
            t0 = time.time()
            X_craft = runner_lu.invert(A_FF, info["model_path"])
            row["infer_s"] = time.time() - t0
            u_craft = X_craft @ b_uniform
            row["craft_err_vs_scipy"] = float(np.max(np.abs(u_craft - u_scipy)))
            row["craft_err_vs_doolittle"] = float(np.max(np.abs(u_craft - u_dool)))
            row["max_inv_diff_craft_vs_dense"] = float(np.max(np.abs(X_craft - np.linalg.inv(A_FF))))
            print(f"  scalar Doolittle vs scipy: {row['scalar_doolittle_err_vs_scipy']:.3e}", flush=True)
            print(f"  CRAFT (X) vs dense inv: {row['max_inv_diff_craft_vs_dense']:.3e}", flush=True)
            print(f"  CRAFT (X @ b) vs scipy: {row['craft_err_vs_scipy']:.3e}", flush=True)
            print(f"  CRAFT (X @ b) vs scalar Doolittle (X @ b): {row['craft_err_vs_doolittle']:.3e}", flush=True)
        except Exception as e:
            row["status"] = f"err: {e!r}"[:200]
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
        rows.append(row)

    out_csv = os.path.join(out_dir, "fem_craft_diag.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
