"""Full-dataset sweep for the merged LU+direct transformer.

For each circuit: build (if cached, skip) -> run -> record. Mirrors
direct_run_all.py output schema so results_main_lu_direct.csv can be diffed
against results_main_direct.csv directly.
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
PROJECT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    os.path.join(PROJECT, "dataset", "circuit_dataset_rv.jsonl"),
)

FIELDS = [
    "circuit_id", "complexity", "N", "n_free", "n_fixed", "target_node",
    "pred_V_dsl", "ref_V_lu", "ref_V_direct", "truth_V",
    "abs_error", "pass_fail",
    "seq_length", "build_time_s", "run_time_s",
    "n_layers", "d_model",
    "build_status", "run_status", "error",
]


def _model_path_for(cid: str, suffix: str) -> str:
    return os.path.join(HERE, f"model_{cid}_lu_direct{suffix}.bin")


def _build_if_missing(cid: str, out_path: str, v_step, k_levels, force) -> tuple[str, dict | None, float]:
    import build_lu_direct as bm
    t0 = time.time()
    if force and os.path.exists(out_path):
        os.remove(out_path)
        if os.path.exists(out_path + ".slots.json"):
            os.remove(out_path + ".slots.json")
    if os.path.exists(out_path) and os.path.exists(out_path + ".slots.json"):
        return "cached", None, 0.0
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            info = bm.build_for_circuit(
                cid, out_path=out_path, plan_only=False,
                v_step=v_step, k_levels=k_levels,
            )
        return "built", info, time.time() - t0
    except Exception as e:
        return f"build_error: {e!r}", None, time.time() - t0
    finally:
        devnull.close()


def _run_one(cid: str, c: dict, tol: float, v_step, k_levels, suffix, force, use_hull) -> dict:
    from parse import parse_netlist  # type: ignore
    from lu_direct_reference import lu_direct_solve_parsed
    from runner_lu_direct import run_one

    row = {fn: "" for fn in FIELDS}
    row["circuit_id"] = cid
    row["complexity"] = c.get("Complexity", "")
    row["truth_V"] = c["Ground_Truth_Vout"]

    try:
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        pc = parse_netlist(netlist)
        n_free = sum(1 for x in pc.is_fixed if not x)
        n_fixed = sum(1 for x in pc.is_fixed if x)
        row["N"] = pc.num_nodes
        row["n_free"] = n_free
        row["n_fixed"] = n_fixed
        row["target_node"] = target

        try:
            ref_v = lu_direct_solve_parsed(pc, target,
                                           v_step=v_step or 500, k_levels=k_levels or 480)
            row["ref_V_lu"] = f"{ref_v:.4f}"
        except Exception as e:
            log.warning(f"{cid}: ref failed: {e}")

        # cross-check vs direct's numpy-solve reference (apples-to-apples).
        try:
            from direct_reference import direct_solve_parsed  # type: ignore
            d_ref = direct_solve_parsed(pc, target,
                                         v_step=v_step or 500, k_levels=k_levels or 480)
            row["ref_V_direct"] = f"{d_ref:.4f}"
        except Exception as e:
            log.warning(f"{cid}: direct_ref failed: {e}")

        out_path = _model_path_for(cid, suffix)
        bstatus, binfo, btime = _build_if_missing(cid, out_path, v_step, k_levels, force)
        row["build_status"] = bstatus
        row["build_time_s"] = f"{btime:.2f}"
        if binfo:
            row["n_layers"] = binfo["n_layers"]
            row["d_model"] = binfo["d_model"]
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        try:
            status, pred_v, truth_v, elapsed = run_one(
                model_path=out_path, netlist=netlist,
                truth=float(c["Ground_Truth_Vout"]),
                tol=tol, v_step=v_step, k_levels=k_levels,
                verbose=False, use_hull=use_hull,
            )
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"] = f"{abs(pred_v - truth_v):.4f}"
            row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            row["seq_length"] = 2 * pc.num_nodes + 3 * n_free + 4
        except Exception as e:
            row["run_status"] = "run_error"
            row["error"] = repr(e)
            log.warning(f"{cid}: run failed: {e}\n{traceback.format_exc()}")
    except Exception as e:
        row["error"] = repr(e)
        log.error(f"{cid}: unexpected: {e}\n{traceback.format_exc()}")
    return row


def write_rows(rows, out_csv):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Merged LU+direct full-dataset sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--no-hull", action="store_true",
                    help="(default) Use StandardKVCache. See runner_lu_direct.py.")
    ap.add_argument("--hull", action="store_true",
                    help="Force HullKVCache (experimental).")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    use_hull = bool(args.hull) and not args.no_hull
    out_csv = os.path.join(RESULTS_DIR, f"results_main_lu_direct{args.suffix}.csv")
    print(f"[lu_direct] use_hull={use_hull}  v_step={args.v_step}  k_levels={args.k_levels}  out={out_csv}",
          flush=True)

    circuits = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]
    print(f"[lu_direct] {len(circuits)} circuits", flush=True)

    rows = []
    pass_n = fail_n = err_n = 0
    for i, c in enumerate(circuits):
        cid = c["ID"]
        t0 = time.time()
        row = _run_one(cid, c, tol=args.tol, v_step=args.v_step, k_levels=args.k_levels,
                       suffix=args.suffix, force=args.force_rebuild, use_hull=use_hull)
        rows.append(row)
        write_rows(rows, out_csv)
        if row["pass_fail"] == "PASS":
            pass_n += 1
        elif row["pass_fail"] == "FAIL":
            fail_n += 1
        else:
            err_n += 1
        dt = time.time() - t0
        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} N={row['N']:<3} n_f={row['n_free']:<3} "
            f"pred={row['pred_V_dsl']:>8} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>7} [{row['pass_fail']}] {dt:.1f}s",
            flush=True,
        )
    print(
        f"\n[lu_direct] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  total={len(circuits)}  out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
