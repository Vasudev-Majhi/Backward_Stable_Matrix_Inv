"""Re-run n=200 inference using the already-built model on disk.

The build (MILP solve, weight bake) was completed in a prior session and
landed at /tmp/lu_models_hull_n200/model_d74966418552_lu.bin.  This script
skips the build and just times inference.

Writes a JSON status file to /tmp/lu_models_hull_n200/_status.json so the
result is recoverable even if the SSH channel drops.
"""
from __future__ import annotations
import os, sys, time, json, traceback

os.environ.pop("MINV_USE_STANDARD_CACHE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

import numpy as np
import runner_lu

assert runner_lu.USING_HULL, "expected Hull cache active"

n = 200
model_dir = "/tmp/lu_models_hull_n200"
status_path = os.path.join(model_dir, "_status.json")

# Reproduce the same A used at build time: rng seed=42, 5I + 0.3*N(0,1).
rng = np.random.RandomState(42)
A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)

import hashlib
h = hashlib.sha1(A.tobytes()).hexdigest()[:12]
model_path = os.path.join(model_dir, f"model_{h}_lu.bin")
assert os.path.exists(model_path), f"model not on disk: {model_path}"
print(f"n={n} USING_HULL={runner_lu.USING_HULL} model={model_path}", flush=True)

status = {"n": n, "phase": "starting", "elapsed_s": 0.0}
with open(status_path, "w") as f:
    json.dump(status, f)

t0 = time.time()
try:
    X = runner_lu.invert(A, model_path)
    t_infer = time.time() - t0
    err = np.abs(X - np.linalg.inv(A))
    max_err = float(err.max()); med_err = float(np.median(err))
    status = {
        "n": n,
        "phase": "done",
        "infer_s": t_infer,
        "max_err": max_err,
        "med_err": med_err,
        "tokens_per_col": 3 * n + 2,
        "total_tokens": (3 * n + 2) * n,
    }
    print(
        f"infer_s={t_infer:.2f}  max_err={max_err:.3e}  med_err={med_err:.3e}  "
        f"tokens_per_col={3*n+2}  total_tokens={(3*n+2)*n}",
        flush=True,
    )
except Exception as e:  # noqa: BLE001
    t_infer = time.time() - t0
    status = {
        "n": n,
        "phase": "error",
        "infer_s": t_infer,
        "error": f"{type(e).__name__}: {e}",
        "traceback": traceback.format_exc(),
    }
    print(f"FAILED after {t_infer:.2f}s: {type(e).__name__}: {e}", flush=True)

with open(status_path, "w") as f:
    json.dump(status, f, indent=2)
print(f"status -> {status_path}", flush=True)
