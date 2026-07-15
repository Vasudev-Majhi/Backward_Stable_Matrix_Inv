"""Cross-domain CRAFT (LU-direct) benchmark runner.

For each case meta JSON in craft/ieee_benchmarks/netlists/, build a
LU-direct CRAFT model per target node (lu_pipeline/build_lu_direct.py) and run
it (lu_pipeline/runner_lu_direct.py).

Strategy: synthesize a temporary JSONL line per target, point
CRAFT_DATASET at it, then call lu_pipeline's `build_for_circuit(cid)` --
the same well-tested code path used for in-house circuits.

Usage:
  python cross_domain_run.py --all
  python cross_domain_run.py --case karate
  python cross_domain_run.py --case karate --max-targets 10
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import csv
import glob
import json
import os
import random
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
NETLISTS_DIR = os.path.join(PROJECT, "craft", "ieee_benchmarks", "netlists")
MODEL_DIR = os.path.join(HERE, "models", "cross_domain")
RESULTS_DIR = os.path.join(HERE, "results", "cross_domain")
DEFAULT_TOL = 0.05  # volts -- matches the in-house 50 mV tolerance

FIELDS = [
    "case", "target_bus", "target_node",
    "truth_v", "truth_vm_pu",
    "pred_v_lu_direct", "ref_v_numpy_solve",
    "abs_error_v", "rel_error_pct",
    "abs_error_vs_numpy_v", "pass_fail",
    "N", "n_free", "n_fixed", "seq_length",
    "build_status", "build_time_s",
    "run_status", "run_time_s",
    "n_layers", "d_model",
    "error",
]


def _numpy_reference(netlist: str, target_node: int) -> float:
    """Direct numpy solve for ground-truth voltage at target_node."""
    import numpy as np
    from parse import parse_netlist  # type: ignore
    pc = parse_netlist(netlist)
    n = pc.num_nodes
    A = np.zeros((n, n), dtype=np.float64)
    for (a, b, ohms) in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g
        A[b, b] += g
        A[a, b] -= g
        A[b, a] -= g
    fixed = [i for i in range(n) if pc.is_fixed[i]]
    free = [i for i in range(n) if not pc.is_fixed[i]]
    A_FF = A[np.ix_(free, free)]
    A_FP = A[np.ix_(free, fixed)]
    v_fixed = np.array([pc.fixed_voltage[i] for i in fixed], dtype=np.float64)
    v_free = np.linalg.solve(A_FF, -A_FP @ v_fixed)
    return float(v_free[free.index(target_node)])


def _build_via_jsonl(cid: str, netlist: str, target_node: int, truth_v: float,
                     model_path: str, force: bool):
    """Build a CRAFT LU-direct model by writing a 1-line JSONL and calling
    build_for_circuit -- the same code path as the in-house sweep."""
    if force and os.path.exists(model_path):
        os.remove(model_path)
        sc = model_path + ".slots.json"
        if os.path.exists(sc):
            os.remove(sc)
    if os.path.exists(model_path) and os.path.exists(model_path + ".slots.json"):
        return "cached", None

    record = {
        "ID": cid,
        "Netlist": netlist,
        "Target_Node": int(target_node),
        "Ground_Truth_Vout": float(truth_v),
        "Complexity": "cross_domain",
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                       delete=False, encoding="utf-8")
    try:
        tmp.write(json.dumps(record) + "\n")
        tmp.close()
        os.environ["CRAFT_DATASET"] = tmp.name
        os.environ.setdefault("MILP_TIME_LIMIT", "120")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import build_lu_direct as bm
        prev_module_dataset = getattr(bm, "DATASET_PATH", None)
        bm.DATASET_PATH = tmp.name  # module-level constant captured at import
        try:
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                info = bm.build_for_circuit(cid, out_path=model_path,
                                             plan_only=False,
                                             v_step=None, k_levels=None)
            return "built", info
        finally:
            if prev_module_dataset is not None:
                bm.DATASET_PATH = prev_module_dataset
    except Exception as e:
        return f"build_error: {e!r}", None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _run_target(case: str, tgt: dict, force: bool, tol: float,
                 keep_model: bool = False, use_hull: bool = False) -> dict:
    row = {fn: "" for fn in FIELDS}
    row["case"] = case
    bid = int(tgt["target_bus"])
    target_node = int(tgt["target_node"])
    truth_v = float(tgt["truth_v"])
    row["target_bus"] = bid
    row["target_node"] = target_node
    row["truth_v"] = f"{truth_v:.4f}"
    row["truth_vm_pu"] = f"{float(tgt.get('truth_vm_pu', truth_v / 12.0)):.6f}"

    netlist_path = os.path.join(NETLISTS_DIR, tgt["netlist_file"])
    safe = case.replace("/", "_")
    cid = f"{safe}_n{bid:05d}"
    model_path = os.path.join(MODEL_DIR, f"{cid}_lu_direct.bin")

    try:
        with open(netlist_path, encoding="utf-8", errors="replace") as f:
            netlist = f.read()

        from parse import parse_netlist  # type: ignore
        pc = parse_netlist(netlist)
        n_free = sum(1 for x in pc.is_fixed if not x)
        n_fixed = sum(1 for x in pc.is_fixed if x)
        row["N"] = pc.num_nodes
        row["n_free"] = n_free
        row["n_fixed"] = n_fixed
        row["seq_length"] = 2 * pc.num_nodes + 3 * n_free + 4

        try:
            ref_v = _numpy_reference(netlist, target_node)
            row["ref_v_numpy_solve"] = f"{ref_v:.6f}"
        except Exception as e:
            row["ref_v_numpy_solve"] = f"err:{e!r}"
            ref_v = None

        t_b = time.time()
        bstatus, binfo = _build_via_jsonl(cid, netlist, target_node, truth_v,
                                           model_path, force)
        row["build_status"] = bstatus
        row["build_time_s"] = f"{time.time() - t_b:.2f}"
        if binfo:
            row["n_layers"] = binfo.get("n_layers", "")
            row["d_model"] = binfo.get("d_model", "")
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        from runner_lu_direct import run_one
        try:
            status, pred_v, _truth_passed, elapsed = run_one(
                model_path=model_path, netlist=netlist, truth=truth_v,
                tol=tol, v_step=None, k_levels=None,
                verbose=False, use_hull=use_hull,
            )
            row["pred_v_lu_direct"] = f"{pred_v:.4f}"
            row["abs_error_v"] = f"{abs(pred_v - truth_v):.4f}"
            row["rel_error_pct"] = f"{abs(pred_v - truth_v) / max(abs(truth_v), 0.01) * 100:.2f}"
            if ref_v is not None:
                row["abs_error_vs_numpy_v"] = f"{abs(pred_v - ref_v):.4f}"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            # Pass/fail vs numpy reference (the meaningful comparison: CRAFT
            # should match numpy's solve on the same parsed netlist).  Some
            # legacy meta JSONs store truth_v in non-volt units, so we don't
            # use truth_v for PASS/FAIL.
            err_for_pass = abs(pred_v - ref_v) if ref_v is not None else abs(pred_v - truth_v)
            row["pass_fail"] = "PASS" if err_for_pass <= tol else "FAIL"
        except Exception as e:
            row["run_status"] = "run_error"
            row["error"] = repr(e)
            print(f"  [!] {case} bus{bid}: run failed: {e}", file=sys.stderr, flush=True)
        finally:
            if not keep_model:
                for p in (model_path, model_path + ".slots.json"):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
    except Exception as e:
        row["error"] = repr(e)
        row["run_status"] = "error"
        print(f"  [!] {case} bus{bid}: setup failed: {e}\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)
    return row


def _write(rows: list[dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_case(case: str, force: bool, tol: float, max_targets: int | None,
             seed: int, keep_models: bool = False, use_hull: bool = False) -> dict:
    meta_path = os.path.join(NETLISTS_DIR, f"{case}_meta.json")
    if not os.path.exists(meta_path):
        print(f"[{case}] no meta.json, skipping", flush=True)
        return {"case": case, "n": 0, "pass": 0, "fail": 0, "err": 0}
    with open(meta_path) as f:
        meta = json.load(f)
    targets = meta.get("targets", [])
    if max_targets is not None and len(targets) > max_targets:
        rng = random.Random(seed)
        targets = rng.sample(targets, max_targets)
        targets.sort(key=lambda t: t["target_bus"])
    print(f"\n[{case}] running {len(targets)} of {len(meta.get('targets', []))} "
          f"targets (max_targets={max_targets}), tol={tol}V", flush=True)

    suffix = "_hull" if use_hull else ""
    out_csv = os.path.join(RESULTS_DIR, f"results_{case}_lu_direct{suffix}.csv")
    rows: list[dict] = []
    for i, tgt in enumerate(targets):
        t0 = time.time()
        row = _run_target(case, tgt, force, tol, keep_model=keep_models, use_hull=use_hull)
        rows.append(row)
        _write(rows, out_csv)  # incremental save
        dt = time.time() - t0
        pf = row.get("pass_fail") or row.get("run_status") or "?"
        print(
            f"  [{i+1:>3}/{len(targets)}] node{int(row['target_node']):4d}  "
            f"pred={row.get('pred_v_lu_direct','?'):>9}  "
            f"truth={row['truth_v']:>9}  "
            f"err={row.get('abs_error_v','?'):>7}  "
            f"{pf:6}  build={row.get('build_time_s','?'):>5}s  "
            f"total={dt:5.1f}s",
            flush=True,
        )

    n = len(rows)
    pass_n = sum(1 for r in rows if r.get("pass_fail") == "PASS")
    fail_n = sum(1 for r in rows if r.get("pass_fail") == "FAIL")
    err_n = sum(1 for r in rows if r.get("run_status") in ("error", "skipped", "run_error"))
    pct = pass_n / n * 100 if n else 0.0
    print(f"[{case}] {pass_n}/{n} PASS ({pct:.1f}%), {fail_n} FAIL, {err_n} ERR", flush=True)
    return {"case": case, "n": n, "pass": pass_n, "fail": fail_n, "err": err_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", type=str, default=None)
    ap.add_argument("--cases", type=str, default=None, help="comma-separated")
    ap.add_argument("--exclude", type=str, default="",
                    help="comma-separated cases to skip (combined with --all)")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--max-targets", type=int, default=None,
                    help="cap number of target nodes per case (random subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-models", action="store_true",
                    help="don't delete model.bin files after each run (uses ~25 MB/circuit)")
    ap.add_argument("--hull", action="store_true",
                    help="use Hull KV cache for inference (writes results_*_lu_direct_hull.csv)")
    args = ap.parse_args()

    if args.all:
        metas = sorted(glob.glob(os.path.join(NETLISTS_DIR, "*_meta.json")))
        cases = [os.path.basename(m).replace("_meta.json", "") for m in metas]
    elif args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    elif args.case:
        cases = [args.case]
    else:
        ap.error("specify --all, --case, or --cases")

    excludes = {c.strip() for c in args.exclude.split(",") if c.strip()}
    cases = [c for c in cases if c not in excludes]

    print(f"[cross_domain] {len(cases)} cases: {', '.join(cases)}", flush=True)
    summary = []
    for c in cases:
        try:
            summary.append(run_case(c, args.force_rebuild, args.tol,
                                     args.max_targets, args.seed,
                                     keep_models=args.keep_models,
                                     use_hull=args.hull))
        except Exception as e:
            print(f"[{c}] failed: {e}\n{traceback.format_exc()}", flush=True)
            summary.append({"case": c, "n": 0, "pass": 0, "fail": 0, "err": -1})

    print("\n=== SUMMARY ===")
    print(f"{'case':<28} {'pass':>5}/{'n':<5} {'fail':>5} {'err':>5}")
    tot_n = tot_p = tot_f = tot_e = 0
    for s in summary:
        print(f"{s['case']:<28} {s['pass']:>5}/{s['n']:<5} {s['fail']:>5} {s['err']:>5}")
        tot_n += s["n"]; tot_p += s["pass"]; tot_f += s["fail"]; tot_e += s["err"]
    print("-" * 60)
    print(f"{'TOTAL':<28} {tot_p:>5}/{tot_n:<5} {tot_f:>5} {tot_e:>5}")

    summary_csv = os.path.join(RESULTS_DIR, "summary_lu_direct.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n", "pass", "fail", "err"])
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {summary_csv}")


if __name__ == "__main__":
    main()
