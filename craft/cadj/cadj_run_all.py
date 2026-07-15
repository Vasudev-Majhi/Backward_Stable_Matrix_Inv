"""C1 — Chebyshev-Accelerated Discrete Jacobi full-dataset sweep with HullKVCache.

Mirrors rbsor_run_all.py / rbsor_run_v01.py for the CADJ algorithm.
Designed for autonomous execution under tmux: incremental CSV checkpoints
after every circuit so partial results survive crashes / kills.

For each circuit:
  1. Compute (lam_min, lam_max) of D^{-1}A_ff via experiments.spectrum
  2. T = auto_T_cadj(N, lam_min, lam_max) or capped at the user floor
  3. Reference: cadj_solve_parsed (mirrors DSL semantics exactly)
  4. Build model_<ID>_cadj.bin if missing
  5. Run via runner.run_circuit with HullKVCache injected, --algorithm cadj
  6. Append row to cadj_results/results_main_cadj[_v01].csv

Usage:
  python cadj_run_all.py                       # full 154 at V_STEP=0.05
  python cadj_run_all.py --v-step 100 --k-levels 2400 --suffix _v01
  python cadj_run_all.py --limit 5             # sanity test
  python cadj_run_all.py --no-hull             # StandardKVCache (debug only)
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

# Add craft/ (parent) and rbsor/ (for jacobi reference comparison) to sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RBSOR_DIR = os.path.join(REPO_ROOT, "rbsor")
for p in (REPO_ROOT, HERE, RBSOR_DIR):
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
    "circuit_id", "complexity", "N", "max_degree", "target_node",
    "lam_min", "lam_max", "T_used",
    "pred_V_dsl", "ref_V_cadj", "jacobi_ref_V", "truth_V",
    "abs_error", "pass_fail", "dsl_matches_ref",
    "seq_length", "run_time_s", "n_layers", "d_model",
    "build_status", "run_status", "error",
]


def _setup_hull_cache(use_hull: bool) -> str:
    import runner
    if use_hull:
        from transformer_vm.attention.hull_cache import HullKVCache
        HullKVCache(1, 1)   # JIT prewarm
        runner.CACHE_CLASS = HullKVCache
        return "HullKVCache"
    from transformer_vm.attention.standard_cache import StandardKVCache
    runner.CACHE_CLASS = StandardKVCache
    return "StandardKVCache"


def _model_path_for(cid: str, suffix: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{cid}_cadj{suffix}.bin")


def _build_if_missing(
    cid: str, T: int, out_path: str, v_step: int | None, k_levels: int | None,
    force: bool,
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
                cid, T=T, out_path=out_path, plan_only=False,
                algorithm="cadj", v_step=v_step, k_levels=k_levels,
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
    from cadj_reference import (
        K_LEVELS as _DEF_K, V_STEP as _DEF_V,
        auto_T_cadj, cadj_solve_parsed,
    )
    from experiments.spectrum import compute_eigenvalue_bounds
    from jacobi_reference import _auto_T as jac_auto_T, jacobi_solve_parsed
    from parse import parse_netlist

    vs_eff = _DEF_V if v_step is None else int(v_step)
    kl_eff = _DEF_K if k_levels is None else int(k_levels)

    row = {fn: "" for fn in FIELDS}
    row["circuit_id"] = cid
    row["complexity"] = c.get("Complexity", "")
    row["truth_V"] = c["Ground_Truth_Vout"]

    try:
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        pc = parse_netlist(netlist)
        row["N"] = pc.num_nodes
        row["max_degree"] = max(pc.degree) if pc.degree else 0
        row["target_node"] = target

        lam_min, lam_max = compute_eigenvalue_bounds(pc)
        row["lam_min"] = f"{lam_min:.6f}"
        row["lam_max"] = f"{lam_max:.6f}"
        T = auto_T_cadj(pc.num_nodes, lam_min, lam_max)
        row["T_used"] = T

        # References (cheap, pure Python).
        try:
            ref_v = cadj_solve_parsed(pc, target, T, lam_min, lam_max,
                                      v_step=vs_eff, k_levels=kl_eff)
            row["ref_V_cadj"] = f"{ref_v:.4f}"
        except Exception as e:
            log.warning(f"{cid}: cadj_ref failed: {e}")
        try:
            jv = jacobi_solve_parsed(pc, target, jac_auto_T(pc.num_nodes))
            row["jacobi_ref_V"] = f"{jv:.4f}"
        except Exception as e:
            log.warning(f"{cid}: jacobi_ref failed: {e}")

        # Build (if missing).
        out_path = _model_path_for(cid, suffix)
        bstatus, binfo = _build_if_missing(cid, T, out_path, v_step, k_levels, force_rebuild)
        row["build_status"] = bstatus
        if binfo:
            row["n_layers"] = binfo["n_layers"]
            row["d_model"] = binfo["d_model"]
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        # Run with whatever cache was injected at module level.
        import runner
        t0 = time.time()
        try:
            status, pred_v, truth, _e = runner.run_circuit(
                cid, MODEL_DIR, T=T, tol=tol, verbose=False,
                algorithm="cadj",
                v_step=v_step, k_levels=k_levels,
                model_path_override=out_path,
            )
            elapsed = time.time() - t0
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"] = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            row["seq_length"] = 2 * pc.num_nodes * (T + 1) + 4
            try:
                ref_f = float(row["ref_V_cadj"]) if row["ref_V_cadj"] else None
                row["dsl_matches_ref"] = (
                    "Y" if (ref_f is not None and abs(pred_v - ref_f) < 0.01) else "N"
                )
            except Exception:
                row["dsl_matches_ref"] = ""
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
    ap = argparse.ArgumentParser(description="C1 — CADJ full-dataset sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--no-hull", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--v-step", type=int, default=None,
                    help="V_STEP in scaled units (500=0.05V default; 100=0.01V)")
    ap.add_argument("--k-levels", type=int, default=None,
                    help="K_LEVELS (480 default; 2400 for V_STEP=100)")
    ap.add_argument("--suffix", type=str, default="",
                    help="Filename suffix for output model + CSV (e.g. '_v01')")
    args = ap.parse_args()

    cache_name = _setup_hull_cache(use_hull=not args.no_hull)
    out_csv = os.path.join(RESULTS_DIR, f"results_main_cadj{args.suffix}.csv")
    print(f"[CADJ] cache={cache_name}  v_step={args.v_step}  k_levels={args.k_levels}  "
          f"out={out_csv}", flush=True)

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]
    print(f"[CADJ] {len(circuits)} circuits", flush=True)

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
            f"lam=[{row['lam_min']},{row['lam_max']}] T={row['T_used']:<5} "
            f"pred={row['pred_V_dsl']:>8} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>7} [{row['pass_fail']}] {dt:.1f}s",
            flush=True,
        )

    print(
        f"\n[CADJ] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"total={len(circuits)}  out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
