"""Experiment C: LU at N=130,150 with HullKVCache.

N=200 is already completed at /tmp/lu_models_hull_n200/.
This script covers N=130 and N=150.

Matrix: same style as the existing sweep (_run_sweep.py):
  A = 5.0*I + 0.3*randn, symmetrised -- well-conditioned random matrix.
Uses HullKVCache (asserts USING_HULL).

Output: exp_c_hull_large_n.csv
Columns: n, max_abs_err, median_abs_err, build_s, infer_s,
         d_model, n_layers, n_params, tokens_per_col, status
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

os.environ.pop("MINV_USE_STANDARD_CACHE", None)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

from build_lu import build_for_matrix  # noqa: E402
import runner_lu  # noqa: E402

assert runner_lu.USING_HULL, "HullKVCache not active — check hull_ext build on this server"

MODEL_DIR = "/tmp/lu_models_hull_large_n"
OUT_CSV = os.path.join(HERE, "exp_c_hull_large_n.csv")

N_VALUES = [130, 150]

FIELDNAMES = ["n", "max_abs_err", "median_abs_err", "build_s", "infer_s",
              "d_model", "n_layers", "n_params", "tokens_per_col", "status"]


def make_matrix(n: int) -> np.ndarray:
    rng = np.random.RandomState(42 + n)
    raw = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)
    return 0.5 * (raw + raw.T)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Append mode so we can resume if interrupted
    csv_f = open(OUT_CSV, "a", newline="", buffering=1)
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    if csv_f.tell() == 0:
        writer.writeheader()
    csv_f.flush()

    print(f"USING_HULL={runner_lu.USING_HULL}", flush=True)
    print(f"{'n':>5} {'max_err':>12} {'med_err':>12} "
          f"{'build_s':>8} {'infer_s':>8} {'d_model':>7}", flush=True)

    for n in N_VALUES:
        try:
            A = make_matrix(n)
            tokens_per_col = 3 * n + 2

            t0 = time.time()
            r = build_for_matrix(A, model_dir=MODEL_DIR)
            build_s = time.time() - t0

            t0 = time.time()
            X = runner_lu.invert(A, r["model_path"])
            infer_s = time.time() - t0

            Xref = np.linalg.inv(A)
            errs = np.abs(X - Xref)
            max_err = float(errs.max())
            med_err = float(np.median(errs))

            row = {
                "n": n, "max_abs_err": max_err, "median_abs_err": med_err,
                "build_s": round(build_s, 3), "infer_s": round(infer_s, 3),
                "d_model": r["d_model"], "n_layers": r["n_layers"],
                "n_params": r["n_params"], "tokens_per_col": tokens_per_col,
                "status": "ok",
            }
            writer.writerow(row)
            csv_f.flush()
            print(f"{n:>5} {max_err:>12.3e} {med_err:>12.3e} "
                  f"{build_s:>8.2f} {infer_s:>8.2f} {r['d_model']:>7}", flush=True)

        except Exception as exc:
            row = {"n": n, "status": f"error:{exc}",
                   "max_abs_err": "", "median_abs_err": "", "build_s": "",
                   "infer_s": "", "d_model": "", "n_layers": "",
                   "n_params": "", "tokens_per_col": ""}
            writer.writerow(row)
            csv_f.flush()
            print(f"{n:>5}  ERROR: {exc}", flush=True)

    csv_f.close()
    print(f"\nDone. Results saved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
