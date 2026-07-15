"""D1 — Direct Sensitivity Solver full-dataset sweep.

For each circuit builds model_<ID>_direct.bin (once) and runs inference.
Build is <1s per circuit (MILP on 4 layers + 10 fetches).
Inference is sub-second (no iteration tokens — just 2N+4 tokens total).

Usage:
  python direct_run_all.py
  python direct_run_all.py --limit 5             # sanity test
  python direct_run_all.py --v-step 100 --k-levels 2400 --suffix _v01
  python direct_run_all.py --no-hull
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
MODEL_DIR = REPO_ROOT
RESULTS_DIR = os.path.join(REPO_ROOT, "cadj_results")

FIELDS = [
    "circuit_id", "complexity", "N", "n_fixed", "target_node",
    "pred_V_dsl", "ref_V_direct", "truth_V",
    "abs_error", "pass_fail",
    "seq_length", "run_time_s", "n_layers", "d_model",
    "build_status", "run_status", "error",
]


def _setup_hull_cache(use_hull: bool) -> str:
    import runner
    if use_hull:
        from transformer_vm.attention.hull_cache import HullKVCache
        HullKVCache(1, 1)
        runner.CACHE_CLASS = HullKVCache
        return "HullKVCache"
    from transformer_vm.attention.standard_cache import StandardKVCache
    runner.CACHE_CLASS = StandardKVCache
    return "StandardKVCache"


def _model_path_for(cid: str, suffix: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{cid}_direct{suffix}.bin")


def _build_if_missing(
    cid: str, out_path: str, v_step: int | None, k_levels: int | None, force: bool,
) -> tuple[str, dict | None]:
    import build as build_mod
    if force and os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(out_path):
        return "cached", None
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            info = build_mod.build_for_circuit(
                cid, T=None, out_path=out_path, plan_only=False,
                algorithm="direct", v_step=v_step, k_levels=k_levels,
            )
        return "built", info
    except Exception as e:
        return f"build_error: {e}", None
    finally:
        devnull.close()


def _run_one(
    cid: str, c: dict, tol: float, v_step: int | None, k_levels: int | None,
    suffix: str, force_rebuild: bool,
) -> dict:
    from direct_reference import direct_solve_parsed
    from parse import parse_netlist

    row = {fn: "" for fn in FIELDS}
    row["circuit_id"] = cid
    row["complexity"] = c.get("Complexity", "")
    row["truth_V"] = c["Ground_Truth_Vout"]

    try:
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        pc = parse_netlist(netlist)
        row["N"] = pc.num_nodes
        row["n_fixed"] = sum(1 for x in pc.is_fixed if x)
        row["target_node"] = target

        # Reference (cheap numpy solve).
        try:
            ref_v = direct_solve_parsed(pc, target, v_step=v_step or 500, k_levels=k_levels or 480)
            row["ref_V_direct"] = f"{ref_v:.4f}"
        except Exception as e:
            log.warning(f"{cid}: direct_ref failed: {e}")

        # Build.
        out_path = _model_path_for(cid, suffix)
        bstatus, binfo = _build_if_missing(cid, out_path, v_step, k_levels, force_rebuild)
        row["build_status"] = bstatus
        if binfo:
            row["n_layers"] = binfo["n_layers"]
            row["d_model"] = binfo["d_model"]
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        # Run.
        import runner
        t0 = time.time()
        try:
            status, pred_v, truth, _e = runner.run_circuit(
                cid, MODEL_DIR, T=None, tol=tol, verbose=False,
                algorithm="direct",
                v_step=v_step, k_levels=k_levels,
                model_path_override=out_path,
            )
            elapsed = time.time() - t0
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"] = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            row["seq_length"] = 2 * pc.num_nodes + 4
        except Exception as e:
            row["run_status"] = "run_error"
            row["error"] = repr(e)
            log.warning(f"{cid}: run failed: {e}\n{traceback.format_exc()}")
    except Exception as e:
        row["error"] = repr(e)
        log.error(f"{cid}: unexpected: {e}\n{traceback.format_exc()}")
    return row


def write_rows(rows: list[dict], out_csv: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="D1 — Direct solver full-dataset sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--no-hull", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    cache_name = _setup_hull_cache(use_hull=not args.no_hull)
    out_csv = os.path.join(RESULTS_DIR, f"results_main_direct{args.suffix}.csv")
    print(f"[DIRECT] cache={cache_name}  v_step={args.v_step}  k_levels={args.k_levels}  "
          f"out={out_csv}", flush=True)

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]
    print(f"[DIRECT] {len(circuits)} circuits", flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0

    for i, c in enumerate(circuits):
        cid = c["ID"]
        t0 = time.time()
        row = _run_one(
            cid, c, tol=args.tol,
            v_step=args.v_step, k_levels=args.k_levels,
            suffix=args.suffix, force_rebuild=args.force_rebuild,
        )
        rows.append(row)
        write_rows(rows, out_csv)

        if row["pass_fail"] == "PASS": pass_n += 1
        elif row["pass_fail"] == "FAIL": fail_n += 1
        else: err_n += 1

        dt = time.time() - t0
        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} N={row['N']:<3} "
            f"pred={row['pred_V_dsl']:>8} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>7} [{row['pass_fail']}] {dt:.1f}s",
            flush=True,
        )

    print(
        f"\n[DIRECT] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"total={len(circuits)}  out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
