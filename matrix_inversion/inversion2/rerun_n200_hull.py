"""Fresh N=200 hull-KV build+infer with a clean model directory.

Forces a genuine re-build by using /tmp/lu_models_n200_rerun (a directory
that does not exist from any previous sweep), ruling out any cached-model
artifact that may have caused the anomalously low build time in lu_interval5.

Run on server:
  ~/craft_release/venv/bin/python rerun_n200_hull.py

Output: results/lu_n200_rerun.csv (single row)
"""
from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
MODEL_DIR = "/tmp/lu_models_n200_rerun"          # fresh dir — no cached weights
OUT_CSV = os.path.join(RESULTS_DIR, "lu_n200_rerun.csv")
TIMEOUT_S = 36000                                 # 10-hour hard limit

FIELDS = [
    "n", "cache",
    "t1_build_s", "t1_infer_s",
    "max_abs_err", "median_abs_err",
    "t1_d_model", "t1_n_layers", "t1_d_ffn", "t1_n_params",
    "status", "error",
]

CODE = f"""
import os, sys, json, time
import numpy as np
HERE = {HERE!r}
sys.path.insert(0, HERE)
import _bootstrap  # sets up transformer-vm on sys.path
from build_lu import build_for_matrix
import runner_lu

n = 200
rng = np.random.RandomState(42)
A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)

print(f"[n200] build start  cache={{os.environ.get('MINV_USE_STANDARD_CACHE','hull')}}", flush=True)
t0 = time.time()
info = build_for_matrix(A, model_dir={MODEL_DIR!r})
t1_build = time.time() - t0
print(f"[n200] build done   {{t1_build:.1f}}s", flush=True)

print("[n200] infer start", flush=True)
t0 = time.time()
X = runner_lu.invert(A, info["model_path"])
t1_infer = time.time() - t0
print(f"[n200] infer done   {{t1_infer:.1f}}s", flush=True)

err = np.abs(X - np.linalg.inv(A))
print(f"[n200] max_err={{float(err.max()):.2e}}", flush=True)

print("RESULT_JSON:" + json.dumps(dict(
    n=200,
    cache="hull" if runner_lu.USING_HULL else "standard",
    t1_build_s=round(t1_build, 3),
    t1_infer_s=round(t1_infer, 3),
    max_abs_err=float(err.max()),
    median_abs_err=float(np.median(err)),
    t1_d_model=info["d_model"],
    t1_n_layers=info["n_layers"],
    t1_d_ffn=info["d_ffn"],
    t1_n_params=info["n_params"],
)))
"""


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = os.environ.copy()
    env.pop("MINV_USE_STANDARD_CACHE", None)   # hull mode

    print(f"[n200-rerun] model dir : {MODEL_DIR}  (fresh — no prior cache)", flush=True)
    print(f"[n200-rerun] expected  : ~2400s build + ~6500s infer (~2.5 h total)", flush=True)
    print(f"[n200-rerun] start     : {time.strftime('%H:%M:%S')}", flush=True)

    t_wall = time.time()
    try:
        cp = subprocess.run(
            [sys.executable, "-u", "-c", CODE],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        print(cp.stdout, flush=True)
        if cp.stderr.strip():
            print("STDERR (last 3000 chars):\n" + cp.stderr[-3000:], flush=True)

        line = next(
            (l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON:")), None
        )
        if line:
            result: dict = json.loads(line[len("RESULT_JSON:"):])
            result["status"] = "ok"
            result["error"] = ""
        else:
            result = {
                "n": 200, "cache": "hull",
                "status": "failed",
                "error": f"no RESULT_JSON; exit={cp.returncode}",
            }
    except subprocess.TimeoutExpired:
        result = {"n": 200, "cache": "hull", "status": "timeout",
                  "error": "10h timeout exceeded"}
    except Exception as exc:
        result = {"n": 200, "cache": "hull", "status": "exception", "error": repr(exc)}

    elapsed = time.time() - t_wall
    print(f"\n[n200-rerun] total wall time: {elapsed:.0f}s", flush=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: result.get(k, "") for k in FIELDS})

    print(f"[n200-rerun] wrote {OUT_CSV}", flush=True)
    print(f"[n200-rerun] status  : {result.get('status')}", flush=True)
    if result.get("status") == "ok":
        print(f"[n200-rerun] build_s : {result.get('t1_build_s')}", flush=True)
        print(f"[n200-rerun] infer_s : {result.get('t1_infer_s')}", flush=True)
        print(f"[n200-rerun] max_err : {result.get('max_abs_err'):.2e}", flush=True)


if __name__ == "__main__":
    main()
