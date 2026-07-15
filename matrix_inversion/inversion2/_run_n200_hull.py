"""One-shot n=200 LU inversion under HullKVCache, with build + infer timing."""
from __future__ import annotations
import os, sys, time

os.environ.pop("MINV_USE_STANDARD_CACHE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

import numpy as np
from build_lu import build_for_matrix
import runner_lu

assert runner_lu.USING_HULL, "expected Hull cache active"

n = 200
os.makedirs("/tmp/lu_models_hull_n200", exist_ok=True)
rng = np.random.RandomState(42)
A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)

print(f"n={n}: USING_HULL={runner_lu.USING_HULL}", flush=True)

t0 = time.time()
r = build_for_matrix(A, model_dir="/tmp/lu_models_hull_n200")
t_build = time.time() - t0
print(f"build_s={t_build:.2f}  d_model={r['d_model']}  n_layers={r['n_layers']}  "
      f"d_ffn={r['d_ffn']}  n_params={r['n_params']:,}", flush=True)

t0 = time.time()
X = runner_lu.invert(A, r["model_path"])
t_infer = time.time() - t0

err = np.abs(X - np.linalg.inv(A))
max_err = float(err.max())
med_err = float(np.median(err))
print(f"infer_s={t_infer:.2f}  max_err={max_err:.3e}  med_err={med_err:.3e}  "
      f"tokens_per_col={3*n+2}  total_tokens={(3*n+2)*n}", flush=True)
