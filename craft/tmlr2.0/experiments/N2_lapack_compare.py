"""N2: LAPACK bit-exact head-to-head.

For each (N, kappa) in a small grid, compare CRAFT's compiled-transformer
inversion against:
  (a) Python reference forward+back substitution starting from the SAME
      L, U produced by lu_factor.doolittle().  This is the bit-exact
      claim: identical L, U inputs → identical X outputs under IEEE 754.
  (b) scipy.linalg.solve(A, I) and numpy.linalg.inv(A)  --- LAPACK GESV
      with partial pivoting, used as a sanity check.

Output:  results/N2_lapack_bitexact.csv

Server invocation:
    cd ~/craft_release/matrix_inversion/inversion2
    ~/craft_release/venv/bin/python3 N2_lapack_compare.py
"""
from __future__ import annotations
import os, sys, csv, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401

import numpy as np
import scipy.linalg as sla

from build_lu import build_for_matrix
import runner_lu
from lu_factor import doolittle


def make_spd(n: int, kappa: float, seed: int) -> np.ndarray:
    """Random SPD with prescribed condition number."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigs = np.geomspace(1.0, kappa, n)
    return (Q * eigs) @ Q.T


def fwdback_doolittle(L: np.ndarray, U: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reference forward+back substitution on (L, U) for each column of B.

    Uses scalar Python loops to mirror CRAFT's per-token IEEE 754 semantics
    exactly: y_i = b_i - sum_{j<i} L[i,j]*y_j ; x_i = (y_i - sum_{j>i} U[i,j]*x_j) / U[i,i]
    """
    n = L.shape[0]
    nrhs = B.shape[1]
    X = np.zeros_like(B)
    for k in range(nrhs):
        b = B[:, k].astype(np.float64)
        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for j in range(i):
                s = s + L[i, j] * y[j]
            y[i] = b[i] - s
        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            s = 0.0
            for j in range(i + 1, n):
                s = s + U[i, j] * x[j]
            x[i] = (y[i] - s) / U[i, i]
        X[:, k] = x
    return X


def main():
    rows = []
    out_path = os.path.join(HERE, "results", "N2_lapack_bitexact.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs("/tmp/N2_models", exist_ok=True)
    seed_base = 0xCAFE
    grid = [(n, k) for n in (5, 10, 15, 20) for k in (1.0, 10.0, 100.0, 1000.0)]

    fields = [
        "n", "kappa_target", "kappa_actual",
        "max_byte_diff_craft_vs_doolittle",  # THE bit-exact metric
        "max_abs_diff_craft_vs_doolittle",
        "max_abs_diff_craft_vs_scipy_solve",
        "max_abs_diff_craft_vs_numpy_inv",
        "max_abs_diff_doolittle_vs_scipy_solve",
        "craft_build_s", "craft_infer_s",
        "status", "error",
    ]
    for (n, k) in grid:
        row = {f: "" for f in fields}
        row["n"] = n
        row["kappa_target"] = k
        try:
            A = make_spd(n, k, seed_base + n * 1000 + int(k))
            row["kappa_actual"] = float(np.linalg.cond(A))

            # CRAFT
            t0 = time.time()
            info = build_for_matrix(A, model_dir="/tmp/N2_models")
            row["craft_build_s"] = time.time() - t0

            t0 = time.time()
            X_craft = runner_lu.invert(A, info["model_path"])
            row["craft_infer_s"] = time.time() - t0

            # Doolittle reference (same L, U scalar IEEE 754)
            L, U = doolittle(A)
            I = np.eye(n, dtype=np.float64)
            X_dool = fwdback_doolittle(L, U, I)

            # LAPACK references
            X_scipy = sla.solve(A, I, assume_a="gen")
            X_numpy = np.linalg.inv(A)

            # Compare byte-for-byte: convert both to bytes and check raw memory diff
            craft_bytes = X_craft.astype(np.float64).tobytes()
            dool_bytes = X_dool.tobytes()
            byte_eq = (craft_bytes == dool_bytes)
            row["max_byte_diff_craft_vs_doolittle"] = 0 if byte_eq else int(
                max(abs(a - b) for a, b in zip(craft_bytes, dool_bytes))
            )

            row["max_abs_diff_craft_vs_doolittle"] = float(np.max(np.abs(X_craft - X_dool)))
            row["max_abs_diff_craft_vs_scipy_solve"] = float(np.max(np.abs(X_craft - X_scipy)))
            row["max_abs_diff_craft_vs_numpy_inv"] = float(np.max(np.abs(X_craft - X_numpy)))
            row["max_abs_diff_doolittle_vs_scipy_solve"] = float(np.max(np.abs(X_dool - X_scipy)))
            row["status"] = "ok"
            print(f"n={n:<3} k={k:>6.0f} byte_diff_vs_doolittle={row['max_byte_diff_craft_vs_doolittle']} "
                  f"max_abs_vs_dool={row['max_abs_diff_craft_vs_doolittle']:.2e} "
                  f"max_abs_vs_scipy={row['max_abs_diff_craft_vs_scipy_solve']:.2e}", flush=True)
        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)[:200]
            print(f"n={n} k={k} ERROR: {e}", flush=True)
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
