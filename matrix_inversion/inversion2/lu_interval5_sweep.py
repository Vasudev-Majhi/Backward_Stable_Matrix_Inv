"""LU sweep at N intervals of 5 with both Standard and Hull KV cache.

Reproduces and extends Tables 7 and 9 of the paper using current CRAFT
(matrix_inversion/inversion2/build_lu.py) on the same well-conditioned
random SPD matrices used previously (A = 5*I + 0.3*N(0,1)).

Output: results/lu_interval5.csv  (one row per (n, cache) tuple).

Usage:
  python lu_interval5_sweep.py                       # N=5,10,...,200
  python lu_interval5_sweep.py --n-max 130           # cap at 130
  python lu_interval5_sweep.py --skip-standard       # only Hull
  python lu_interval5_sweep.py --skip-hull           # only standard
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "lu_interval5.csv")
MODEL_DIR = "/tmp/lu_models_interval5"

FIELDS = [
    "n", "cache",
    "t1_build_s", "t1_infer_s",
    "t2_build_s", "t2_infer_s",
    "max_abs_err", "median_abs_err",
    "tokens_per_col", "t1_d_model", "t1_n_layers", "t1_d_ffn", "t1_n_params",
    "t2_d_model", "t2_n_params",
    "status", "error",
    # Legacy column aliases retained for downstream scripts
    "build_s", "infer_s", "d_model", "n_layers", "d_ffn", "n_params",
]


def make_A(n: int, seed: int = 42) -> np.ndarray:
    """Same well-conditioned generator the previous tables used."""
    rng = np.random.RandomState(seed)
    return 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)


def run_one(n: int, use_hull: bool) -> dict:
    """Run a single (n, cache) experiment in a fresh subprocess.

    Subprocess isolation matters: USING_HULL is set at runner_lu import time
    based on env var, so we can't toggle it within one process.
    """
    env = os.environ.copy()
    if use_hull:
        env.pop("MINV_USE_STANDARD_CACHE", None)
    else:
        env["MINV_USE_STANDARD_CACHE"] = "1"

    code = f"""
import os, sys, json, time
import numpy as np
HERE = {HERE!r}
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401
from build_lu import build_for_matrix
import runner_lu

n = {n}
seed = 42
rng = np.random.RandomState(seed)
A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)

# ── T1: build + run ───────────────────────────────────────────────────────────
t0 = time.time()
info = build_for_matrix(A, model_dir={MODEL_DIR!r})
t1_build = time.time() - t0

t0 = time.time()
X = runner_lu.invert(A, info["model_path"])
t1_infer = time.time() - t0

err = np.abs(X - np.linalg.inv(A))

# ── T2: build a synthetic readout (random A_FP, k=2 sources, S=-X@A_FP) ──────
t2_build = -1.0
t2_infer = -1.0
t2_d_model = -1
t2_n_params = -1
try:
    # Add CADJ and CLAUDE_FILES paths only now, AFTER T1 done, to avoid any
    # import-order interaction with build_lu / runner_lu.
    for _p in ("craft/cadj",
                "craft"):
        if _p not in sys.path:
            sys.path.append(_p)
    k = 2
    A_FP = 0.3 * rng.randn(n, k)
    S = -X @ A_FP  # n x k sensitivity matrix
    from parse import ParsedCircuit  # type: ignore
    from direct_interpreter import DirectCircuitMachine  # type: ignore
    from transformer_vm.scheduler.milp import milp_schedule  # type: ignore
    from transformer_vm.model.weights import build_model  # type: ignore
    N_total = n + k
    is_fixed = [False]*n + [True]*k
    fixed_voltage = [0.0]*n + [12.0, 8.0][:k]
    resistors = [(i, i+1, 1.0) for i in range(N_total-1)]
    pc = ParsedCircuit(num_nodes=N_total, is_fixed=is_fixed,
                        fixed_voltage=fixed_voltage, resistors=resistors)
    machine = DirectCircuitMachine(pc, target_node=0, S_override=S)
    t0 = time.time()
    pg, _ = machine.build()
    _sched = milp_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
    model, all_tokens, _, _ = build_model(program_graph=pg)
    t2_build = time.time() - t0
    t2_d_model = model.tok.weight.shape[1]
    t2_n_params = sum(p.numel() for p in model.parameters())
    import torch
    model.eval()
    x = torch.zeros(t2_d_model, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad():
        _ = model.head(x)
    t2_infer = time.time() - t0
except Exception as _e:
    print("T2_FAIL:" + repr(_e), flush=True)

out = dict(
    cache="hull" if runner_lu.USING_HULL else "standard",
    t1_build_s=round(t1_build, 3),
    t1_infer_s=round(t1_infer, 3),
    t2_build_s=round(t2_build, 3) if t2_build >= 0 else None,
    t2_infer_s=round(t2_infer, 6) if t2_infer >= 0 else None,
    max_abs_err=float(err.max()),
    median_abs_err=float(np.median(err)),
    tokens_per_col=3 * n + 2,
    t1_d_model=info["d_model"],
    t1_n_layers=info["n_layers"],
    t1_d_ffn=info["d_ffn"],
    t1_n_params=info["n_params"],
    t2_d_model=t2_d_model if t2_d_model > 0 else None,
    t2_n_params=t2_n_params if t2_n_params > 0 else None,
    # Legacy aliases
    build_s=round(t1_build, 3),
    infer_s=round(t1_infer, 3),
    d_model=info["d_model"],
    n_layers=info["n_layers"],
    d_ffn=info["d_ffn"],
    n_params=info["n_params"],
)
print("RESULT_JSON:" + json.dumps(out))
"""
    try:
        cp = subprocess.run([sys.executable, "-u", "-c", code], env=env,
                             capture_output=True, text=True, timeout=14400)
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if not line:
            err_tail = "\n".join(cp.stderr.splitlines()[-12:])
            return {"n": n, "cache": "hull" if use_hull else "standard",
                    "status": "failed", "error": f"no result; stderr tail: {err_tail!r}"}
        import json as _json
        d = _json.loads(line[len("RESULT_JSON:"):])
        d["n"] = n
        d["status"] = "ok"
        return d
    except subprocess.TimeoutExpired:
        return {"n": n, "cache": "hull" if use_hull else "standard",
                "status": "timeout", "error": "4h timeout exceeded"}
    except Exception as e:
        return {"n": n, "cache": "hull" if use_hull else "standard",
                "status": "exception", "error": repr(e) + "\n" + traceback.format_exc()}


def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=200)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--skip-standard", action="store_true")
    ap.add_argument("--skip-hull", action="store_true")
    ap.add_argument("--standard-cap", type=int, default=130,
                    help="don't run standard cache above this N (it times out)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel workers (each spawns its own subprocess)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_values = list(range(args.n_min, args.n_max + 1, args.step))
    rows: list[dict] = []
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({k: r.get(k, "") for k in FIELDS})
    done = {(int(r["n"]), r["cache"]) for r in rows if r.get("status") == "ok"}

    lock = threading.Lock()

    def flush():
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # Build task list: (n, use_hull) for all pending experiments
    tasks: list[tuple[int, bool]] = []
    for n in n_values:
        for use_hull in ([True] if args.skip_standard else
                         [False] if args.skip_hull else [False, True]):
            cache_name = "hull" if use_hull else "standard"
            if not use_hull and n > args.standard_cap:
                continue
            if (n, cache_name) in done:
                print(f"{n:>4} {cache_name:>8}    cached", flush=True)
                continue
            tasks.append((n, use_hull))

    print(f"\n{len(tasks)} tasks to run with {args.workers} workers\n"
          f"{'N':>4} {'cache':>8} {'T1b':>7} {'T1i':>9} "
          f"{'T2b':>6} {'T2i':>9} {'err':>11} {'d1':>5} {'d2':>4} {'status':>8}",
          flush=True)

    start_times: dict[tuple[int, bool], float] = {}

    def _run(n: int, use_hull: bool) -> dict:
        start_times[(n, use_hull)] = time.time()
        return run_one(n, use_hull)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, n, h): (n, h) for n, h in tasks}
        for fut in as_completed(futures):
            n, use_hull = futures[fut]
            cache_name = "hull" if use_hull else "standard"
            dt = time.time() - start_times.get((n, use_hull), time.time())
            try:
                r = fut.result()
            except Exception as e:
                r = {"n": n, "cache": cache_name, "status": "exception", "error": repr(e)}
            with lock:
                rows.append({k: r.get(k, "") for k in FIELDS})
                flush()
            status = r.get("status", "?")
            mae = r.get("max_abs_err")
            mae_str = f"{mae:.2e}" if isinstance(mae, (int, float)) else ""
            print(
                f"{n:>4} {cache_name:>8} "
                f"T1b={str(r.get('t1_build_s','?')):>6} "
                f"T1i={str(r.get('t1_infer_s','?')):>6} "
                f"T2b={str(r.get('t2_build_s','?')):>6} "
                f"T2i={str(r.get('t2_infer_s','?')):>8} "
                f"err={mae_str:>9} "
                f"d1={str(r.get('t1_d_model','?')):>5} "
                f"d2={str(r.get('t2_d_model','?')):>4} "
                f"{status:>8}  ({dt:.0f}s)",
                flush=True,
            )

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
