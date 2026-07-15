"""154-circuit sweep with StandardKVCache for both Tx1 (LU inversion) and Tx2 (readout inference).

Tx1: forces MINV_USE_STANDARD_CACHE=1 so runner_lu.py uses StandardKVCache.
Tx2: uses _server_run_idea2.py (StandardKVCache, unchanged).

Stops the sweep if total elapsed time exceeds --max-minutes (default 20).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import traceback

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force Standard KV cache for Tx1 (runner_lu.py subprocess inherits this env var)
os.environ["MINV_USE_STANDARD_CACHE"] = "1"

import _bootstrap  # noqa: F401

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl")
DATASET_PATH = os.environ.get("CRAFT_DATASET", _DEFAULT_DATASET)
MODEL_DIR    = _CLAUDE_FILES
RESULTS_DIR  = os.path.join(_CLAUDE_FILES, "cadj_results")

FIELDS = [
    "circuit_id", "complexity", "N", "n_free", "n_fixed", "target_node",
    "pred_V", "truth_V", "abs_error", "pass_fail",
    "lu_build_s", "lu_infer_s", "readout_build_s", "infer_s", "n_tokens",
    "tx1_cache", "tx2_cache", "build_status", "error",
]


def _write_rows(rows: list[dict], out_csv: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse
    from _server_build_idea2 import build_for_circuit
    from _server_run_idea2 import run_circuit_idea2

    ap = argparse.ArgumentParser(description="Idea2: StandardKVCache 154-circuit sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true", default=True)
    ap.add_argument("--no-force-rebuild", dest="force_rebuild", action="store_false")
    ap.add_argument("--tol", type=float, default=0.075)
    ap.add_argument("--suffix", type=str, default="_standard")
    ap.add_argument("--max-minutes", type=float, default=20.0,
                    help="Stop sweep if total elapsed exceeds this many minutes")
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    args = ap.parse_args()

    max_seconds = args.max_minutes * 60.0

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]

    out_csv = os.path.join(RESULTS_DIR, f"results_idea2{args.suffix}.csv")
    print(f"[STD SWEEP] {len(circuits)} circuits  tol={args.tol}  force_rebuild={args.force_rebuild}  max_minutes={args.max_minutes}", flush=True)
    print(f"[STD SWEEP] Tx1=StandardKVCache  Tx2=StandardKVCache  out={out_csv}", flush=True)
    print(flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0
    sweep_t0 = time.time()

    for i, c in enumerate(circuits):
        elapsed = time.time() - sweep_t0
        if elapsed > max_seconds:
            print(f"\n[STD SWEEP] TIME LIMIT {args.max_minutes:.0f}min reached after {i} circuits. Stopping.", flush=True)
            break

        cid = c["ID"]
        row = {fn: "" for fn in FIELDS}
        row["circuit_id"]  = cid
        row["complexity"]  = c.get("Complexity", "")
        row["truth_V"]     = c["Ground_Truth_Vout"]
        row["target_node"] = c["Target_Node"]
        row["tx1_cache"]   = "standard"
        row["tx2_cache"]   = "standard"
        t0 = time.time()

        try:
            model_path  = os.path.join(MODEL_DIR, f"model_{cid}_idea2.bin")
            needs_build = not os.path.exists(model_path) or args.force_rebuild

            if needs_build:
                info = build_for_circuit(
                    cid, force=args.force_rebuild,
                    v_step=args.v_step, k_levels=args.k_levels,
                )
                row["build_status"]    = "built"
                row["N"]               = info.get("N", "")
                row["n_free"]          = info.get("n_free", "")
                row["n_fixed"]         = info.get("n_fixed", "")
                row["lu_build_s"]      = f"{info.get('lu_build_s', 0.0):.3f}"
                row["lu_infer_s"]      = f"{info.get('lu_infer_s', 0.0):.3f}"
                row["readout_build_s"] = f"{info.get('readout_build_s', 0.0):.3f}"
            else:
                row["build_status"] = "cached"

            status, pred_v, truth, infer_s, n_tokens = run_circuit_idea2(
                cid, tol=args.tol,
                v_step=args.v_step, k_levels=args.k_levels,
            )
            row["pred_V"]     = f"{pred_v:.4f}"
            row["abs_error"]  = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"]  = "PASS" if status == "PASS" else "FAIL"
            row["infer_s"]    = f"{infer_s:.3f}"
            row["n_tokens"]   = n_tokens

            if status == "PASS":
                pass_n += 1
            else:
                fail_n += 1

        except Exception as e:
            row["error"]        = repr(e)
            row["pass_fail"]    = "ERROR"
            row["build_status"] = "error"
            err_n += 1
            log.error("%s: %s\n%s", cid, e, traceback.format_exc())

        rows.append(row)
        _write_rows(rows, out_csv)

        dt = time.time() - t0
        total_elapsed = time.time() - sweep_t0
        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} "
            f"pred={row.get('pred_V','--'):>8} truth={float(row['truth_V']):>8.4f} "
            f"err={row.get('abs_error','--'):>7} [{row['pass_fail']}]  "
            f"lu_bld={row.get('lu_build_s','--'):>7}s "
            f"lu_inf={row.get('lu_infer_s','--'):>7}s "
            f"r2_bld={row.get('readout_build_s','--'):>6}s "
            f"infer={row.get('infer_s','--'):>7}s "
            f"toks={row.get('n_tokens','--')} total={dt:.1f}s elapsed={total_elapsed:.0f}s",
            flush=True,
        )

    total = time.time() - sweep_t0
    print(flush=True)
    print(
        f"[STD SWEEP DONE]  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"ran={len(rows)}/{len(circuits)}  total_time={total:.0f}s  out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
