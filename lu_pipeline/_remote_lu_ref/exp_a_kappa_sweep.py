"""Experiment A: LU condition number sweep.

For each (N, kappa) pair builds a random SPD matrix with exactly that condition
number, builds the LU transformer, runs inference, and records error metrics.

Output: exp_a_kappa_sweep.csv
Columns: n, kappa, max_abs_err, median_abs_err, build_s, infer_s, d_model, n_params, status
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

from build_lu import build_for_matrix  # noqa: E402
from runner_lu import invert  # noqa: E402

MODEL_DIR = "/tmp/lu_models_kappa_sweep"
OUT_CSV = os.path.join(HERE, "exp_a_kappa_sweep.csv")

N_VALUES = [10, 20, 50]
KAPPA_VALUES = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

FIELDNAMES = ["n", "kappa", "max_abs_err", "median_abs_err",
              "build_s", "infer_s", "d_model", "n_params", "status"]


def make_spd_kappa(n: int, kappa: float, seed: int = 0) -> np.ndarray:
    """N×N SPD matrix with condition number exactly kappa.

    Eigenvalues are geometrically spaced: [1, kappa^(1/(n-1)), ..., kappa].
    Q is a random orthogonal matrix from QR decomposition.
    """
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    if n == 1:
        eigvals = np.array([1.0])
    elif kappa == 1:
        eigvals = np.ones(n)
    else:
        r = kappa ** (1.0 / (n - 1))
        eigvals = r ** np.arange(n, dtype=np.float64)
    A = (Q * eigvals) @ Q.T
    # Symmetrise to kill any floating point asymmetry
    A = 0.5 * (A + A.T)
    return A


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    csv_f = open(OUT_CSV, "w", newline="", buffering=1)
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    writer.writeheader()
    csv_f.flush()

    print(f"{'n':>4} {'kappa':>7} {'max_err':>12} {'med_err':>12} "
          f"{'build_s':>8} {'infer_s':>8} {'d_model':>7}", flush=True)

    for n in N_VALUES:
        for kappa in KAPPA_VALUES:
            seed = n * 10000 + int(kappa)
            try:
                A = make_spd_kappa(n, kappa, seed=seed)

                # Sanity check: verify actual condition number
                actual_kappa = float(np.linalg.cond(A))

                t0 = time.time()
                r = build_for_matrix(A, model_dir=MODEL_DIR)
                build_s = time.time() - t0

                t0 = time.time()
                X = invert(A, r["model_path"])
                infer_s = time.time() - t0

                Xref = np.linalg.inv(A)
                errs = np.abs(X - Xref)
                max_err = float(errs.max())
                med_err = float(np.median(errs))

                row = {
                    "n": n, "kappa": kappa,
                    "max_abs_err": max_err, "median_abs_err": med_err,
                    "build_s": round(build_s, 3), "infer_s": round(infer_s, 3),
                    "d_model": r["d_model"], "n_params": r["n_params"],
                    "status": "ok",
                }
                writer.writerow(row)
                csv_f.flush()
                print(f"{n:>4} {kappa:>7} {max_err:>12.3e} {med_err:>12.3e} "
                      f"{build_s:>8.2f} {infer_s:>8.2f} {r['d_model']:>7}", flush=True)

            except Exception as exc:
                row = {"n": n, "kappa": kappa, "status": f"error:{exc}",
                       "max_abs_err": "", "median_abs_err": "",
                       "build_s": "", "infer_s": "", "d_model": "", "n_params": ""}
                writer.writerow(row)
                csv_f.flush()
                print(f"{n:>4} {kappa:>7}  ERROR: {exc}", flush=True)

    csv_f.close()
    print(f"\nDone. Results saved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
