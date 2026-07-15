"""Run the build+inference LU sweep with HullKVCache forced.

Mirrors _run_sweep.py but does NOT set MINV_USE_STANDARD_CACHE, and asserts
that runner_lu picked the Hull cache.  Writes a CSV summary alongside the
existing standard-cache results so the two can be compared.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

# Make sure the standard-cache fallback is NOT active.
os.environ.pop("MINV_USE_STANDARD_CACHE", None)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

from build_lu import build_for_matrix  # noqa: E402
import runner_lu  # noqa: E402

assert runner_lu.USING_HULL, "expected Hull cache active; check hull_ext build"


def main(out_csv: str = "lu_sweep_hull.csv", n_min: int = 2, n_max: int = 100,
         model_dir: str = "/tmp/lu_models_hull"):
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    fieldnames = ["n", "build_s", "infer_s", "max_abs_err", "median_abs_err",
                  "tokens_per_col", "d_model", "n_layers", "d_ffn", "n_params", "error"]
    csv_f = open(out_csv, "w", newline="", buffering=1)
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()
    csv_f.flush()

    print(f"{'n':>3} {'build_s':>8} {'infer_s':>8} {'max_err':>10} "
          f"{'med_err':>10} {'tokens':>7} {'d_model':>7}", flush=True)
    for n in range(n_min, n_max + 1):
        A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)
        try:
            t0 = time.time()
            r = build_for_matrix(A, model_dir=model_dir)
            t_build = time.time() - t0

            t0 = time.time()
            X = runner_lu.invert(A, r["model_path"])
            t_run = time.time() - t0

            Xref = np.linalg.inv(A)
            err = np.abs(X - Xref)
            max_err = float(np.max(err))
            med_err = float(np.median(err))
            tokens_per_col = 3 * n + 2

            print(f"{n:>3} {t_build:>8.2f} {t_run:>8.2f} {max_err:>10.2e} "
                  f"{med_err:>10.2e} {tokens_per_col:>7} {r['d_model']:>7}",
                  flush=True)
            row = {
                "n": n,
                "build_s": round(t_build, 3),
                "infer_s": round(t_run, 3),
                "max_abs_err": max_err,
                "median_abs_err": med_err,
                "tokens_per_col": tokens_per_col,
                "d_model": r["d_model"],
                "n_layers": r["n_layers"],
                "d_ffn": r["d_ffn"],
                "n_params": r["n_params"],
            }
        except Exception as e:  # pragma: no cover
            print(f"{n:>3} FAILED: {type(e).__name__}: {e}", flush=True)
            row = {"n": n, "error": f"{type(e).__name__}: {e}"}
        writer.writerow(row)
        csv_f.flush()

    csv_f.close()
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
