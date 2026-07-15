"""Step-10 probe of the LU pipeline upper limit.

Tries n = 10, 20, 30, ... up to N_MAX. For each n:
  - First runs MILP plan-only with PLAN_BUDGET_S timeout.
  - If that succeeds and d_model is reasonable, does a full build + inversion
    with FULL_BUDGET_S timeout.
  - Logs one row per n (CSV is line-buffered so partial results survive a kill).

Stops at the first n where plan or full hits a timeout / error / out-of-memory.
"""
from __future__ import annotations

import csv
import os
import resource
import signal
import sys
import time

import numpy as np

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

from build_lu import build_for_matrix  # noqa: E402
from runner_lu import invert  # noqa: E402

STEP = 10
N_MIN = 10
N_MAX = 1000

PLAN_BUDGET_S = 600     # 10 min on MILP alone
FULL_BUDGET_S = 3600    # 1 hour total per n for full build + invert
D_MODEL_LIMIT = 12000   # skip full inversion if d_model exceeds this


class _TO(Exception):
    pass


def _alarm(signum, frame):
    raise _TO()


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(out_csv: str = "lu_step10.csv", model_dir: str = "/tmp/lu_step10"):
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    fields = ["n", "phase", "elapsed_s", "build_s", "infer_s",
              "max_abs_err", "median_abs_err", "tokens_per_col",
              "d_model", "n_layers", "d_ffn", "n_params",
              "peak_rss_mb", "status", "note"]
    f = open(out_csv, "w", newline="", buffering=1)
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    f.flush()

    print(f"{'n':>4} {'phase':>6} {'elapsed':>8} {'build':>8} {'infer':>8} "
          f"{'max_err':>10} {'rss_MB':>9} {'status':>12}  note", flush=True)

    stop = False
    for n in range(N_MIN, N_MAX + 1, STEP):
        if stop:
            break
        A = 5.0 * np.eye(n) + 0.3 * rng.randn(n, n)

        # ---- Phase 1: plan-only ----
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(PLAN_BUDGET_S)
        t0 = time.time()
        plan_d_model = None
        try:
            r = build_for_matrix(A, model_dir=model_dir, plan_only=True)
            plan_d_model = r.get("width") or r.get("d_model")
            t_plan = time.time() - t0
            signal.alarm(0)
            print(f"{n:>4} {'plan':>6} {t_plan:>8.1f} {'-':>8} {'-':>8} "
                  f"{'-':>10} {_peak_rss_mb():>9.0f} {'plan_ok':>12}  "
                  f"d_model={plan_d_model}", flush=True)
            w.writerow({"n": n, "phase": "plan",
                        "elapsed_s": round(t_plan, 2),
                        "d_model": plan_d_model,
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "plan_ok", "note": ""})
            f.flush()
        except _TO:
            t_plan = time.time() - t0
            print(f"{n:>4} {'plan':>6} {t_plan:>8.1f} {'-':>8} {'-':>8} "
                  f"{'-':>10} {_peak_rss_mb():>9.0f} {'plan_TO':>12}  "
                  f"plan exceeded {PLAN_BUDGET_S}s", flush=True)
            w.writerow({"n": n, "phase": "plan",
                        "elapsed_s": round(t_plan, 2),
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "plan_timeout",
                        "note": f"plan exceeded {PLAN_BUDGET_S}s"})
            f.flush()
            stop = True
            break
        except Exception as e:
            t_plan = time.time() - t0
            print(f"{n:>4} {'plan':>6} {t_plan:>8.1f} {'-':>8} {'-':>8} "
                  f"{'-':>10} {_peak_rss_mb():>9.0f} {'plan_ERR':>12}  "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            w.writerow({"n": n, "phase": "plan",
                        "elapsed_s": round(t_plan, 2),
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "plan_error",
                        "note": f"{type(e).__name__}: {e}"})
            f.flush()
            stop = True
            break
        finally:
            signal.alarm(0)

        if plan_d_model and plan_d_model > D_MODEL_LIMIT:
            print(f"==> skipping full at n={n}: d_model={plan_d_model} > "
                  f"{D_MODEL_LIMIT}", flush=True)
            w.writerow({"n": n, "phase": "full",
                        "d_model": plan_d_model,
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "skipped",
                        "note": f"d_model > {D_MODEL_LIMIT}"})
            f.flush()
            continue

        # ---- Phase 2: full build + inversion ----
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(FULL_BUDGET_S)
        t_full0 = time.time()
        try:
            tb0 = time.time()
            r = build_for_matrix(A, model_dir=model_dir)
            t_build = time.time() - tb0
            ti0 = time.time()
            X = invert(A, r["model_path"])
            t_run = time.time() - ti0
            t_full = time.time() - t_full0
            signal.alarm(0)

            err = np.abs(X - np.linalg.inv(A))
            max_err = float(np.max(err))
            med_err = float(np.median(err))

            print(f"{n:>4} {'full':>6} {t_full:>8.1f} {t_build:>8.2f} "
                  f"{t_run:>8.2f} {max_err:>10.2e} "
                  f"{_peak_rss_mb():>9.0f} {'OK':>12}  "
                  f"d_model={r['d_model']} params={r['n_params']:,}",
                  flush=True)
            w.writerow({"n": n, "phase": "full",
                        "elapsed_s": round(t_full, 2),
                        "build_s": round(t_build, 2),
                        "infer_s": round(t_run, 2),
                        "max_abs_err": max_err,
                        "median_abs_err": med_err,
                        "tokens_per_col": 3 * n + 2,
                        "d_model": r["d_model"],
                        "n_layers": r["n_layers"],
                        "d_ffn": r["d_ffn"],
                        "n_params": r["n_params"],
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "ok", "note": ""})
            f.flush()
        except _TO:
            t_full = time.time() - t_full0
            print(f"{n:>4} {'full':>6} {t_full:>8.1f} {'-':>8} {'-':>8} "
                  f"{'-':>10} {_peak_rss_mb():>9.0f} {'full_TO':>12}  "
                  f"exceeded {FULL_BUDGET_S}s", flush=True)
            w.writerow({"n": n, "phase": "full",
                        "elapsed_s": round(t_full, 2),
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "full_timeout",
                        "note": f"exceeded {FULL_BUDGET_S}s"})
            f.flush()
            stop = True
            break
        except MemoryError as e:
            t_full = time.time() - t_full0
            print(f"{n:>4} {'full':>6} {t_full:>8.1f} {'-':>8} {'-':>8} "
                  f"{'-':>10} {_peak_rss_mb():>9.0f} {'OOM':>12}  "
                  f"{e}", flush=True)
            w.writerow({"n": n, "phase": "full",
                        "elapsed_s": round(t_full, 2),
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "oom", "note": str(e)})
            f.flush()
            stop = True
            break
        except Exception as e:
            t_full = time.time() - t_full0
            print(f"{n:>4} {'full':>6} {t_full:>8.1f} {'-':>8} {'-':>8} "
                  f"{'-':>10} {_peak_rss_mb():>9.0f} {'full_ERR':>12}  "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            w.writerow({"n": n, "phase": "full",
                        "elapsed_s": round(t_full, 2),
                        "peak_rss_mb": round(_peak_rss_mb(), 1),
                        "status": "full_error",
                        "note": f"{type(e).__name__}: {e}"})
            f.flush()
            stop = True
            break
        finally:
            signal.alarm(0)

    f.close()
    print("\nWrote " + out_csv, flush=True)


if __name__ == "__main__":
    main()
