"""Bounded Hull-forced sweep n=2..30 with build/infer timings.

Run on the server.  Writes results/lu_sweep_hull_n2_30.csv.  Forces Hull
cache and asserts USING_HULL is True.
"""
from __future__ import annotations
import os, sys, time, csv

os.environ.pop("MINV_USE_STANDARD_CACHE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

import numpy as np
from build_lu import build_for_matrix
import runner_lu

assert runner_lu.USING_HULL, "expected Hull cache active"

os.makedirs("/tmp/lu_models_hull", exist_ok=True)
os.makedirs("results", exist_ok=True)

rng = np.random.RandomState(42)
out_csv = "results/lu_sweep_hull_n2_30.csv"
fieldnames = [
    "n", "build_s", "infer_s", "max_abs_err", "median_abs_err",
    "tokens_per_col", "d_model", "n_layers", "d_ffn", "n_params",
]

cf = open(out_csv, "w", newline="")
w = csv.DictWriter(cf, fieldnames=fieldnames)
w.writeheader()

print(
    f"{'n':>3} {'build_s':>8} {'infer_s':>8} {'max_err':>10} "
    f"{'med_err':>10} {'tokens':>7} {'d_model':>7}",
    flush=True,
)

for n in range(2, 31):
    A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)
    try:
        t0 = time.time()
        r = build_for_matrix(A, model_dir="/tmp/lu_models_hull")
        t_build = time.time() - t0
        t0 = time.time()
        X = runner_lu.invert(A, r["model_path"])
        t_run = time.time() - t0
        err = np.abs(X - np.linalg.inv(A))
        max_err = float(err.max())
        med_err = float(np.median(err))
        print(
            f"{n:>3} {t_build:>8.2f} {t_run:>8.2f} {max_err:>10.2e} "
            f"{med_err:>10.2e} {3*n+2:>7} {r['d_model']:>7}",
            flush=True,
        )
        w.writerow({
            "n": n,
            "build_s": round(t_build, 3),
            "infer_s": round(t_run, 3),
            "max_abs_err": max_err,
            "median_abs_err": med_err,
            "tokens_per_col": 3 * n + 2,
            "d_model": r["d_model"],
            "n_layers": r["n_layers"],
            "d_ffn": r["d_ffn"],
            "n_params": r["n_params"],
        })
        cf.flush()
    except Exception as e:
        print(f"{n:>3} FAILED: {type(e).__name__}: {e}", flush=True)

cf.close()
print()
print(f"Wrote {out_csv}")
