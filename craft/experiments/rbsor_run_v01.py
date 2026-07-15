"""E1V01 — RB-SOR with V_STEP=0.01, omega_cap=1.5, auto-fallback to Jacobi.

Per-circuit decision logic:
  1. If conflicts > 0 (non-bipartite graph) -> use Jacobi
  2. Else run Python rbsor_reference at iter=50 and iter=100; if predictions
     differ by > 1V or |pred| > 24V -> use Jacobi (divergence detector)
  3. Else use RB-SOR with omega = min(omega_opt(rho), 1.5)

Build target:
  - V_STEP = 100 scaled units = 0.01V (5x finer than default 0.05V)
  - K_LEVELS = 2400 (covers 0..23.99V)
  - T = max(500, 50*N)
  - omega_cap = 1.5 (rbsor only)
  - Output models: model_<ID>_<algo>_v01.bin (algo = rbsor or jacobi)

Output CSV: rbsor_results/results_main_rbsor_v01.csv with extra columns:
  solver_used (rbsor|jacobi), omega_used, conflicts, fallback_reason, divergence_pred_50, divergence_pred_100

Hull cache is enabled module-globally before any circuit runs.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

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
RBSOR_DIR = os.path.join(REPO_ROOT, "rbsor")
for p in (REPO_ROOT, RBSOR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
MODEL_DIR = REPO_ROOT
RESULTS_DIR = os.path.join(REPO_ROOT, "rbsor_results")
OUT_CSV = os.path.join(RESULTS_DIR, "results_main_rbsor_v01.csv")

# ── v01 configuration constants ─────────────────────────────────────────────
V01_V_STEP = 100        # 0.01V in scaled units (SCALE = 10000)
V01_K_LEVELS = 2400     # 0..23.99V
V01_T_FLOOR = 500
V01_OMEGA_CAP = 1.5

FIELDS = [
    "circuit_id", "complexity", "N", "max_degree", "target_node",
    "rho_J", "omega_used", "T_used", "n_red", "n_black", "conflicts",
    "solver_used", "fallback_reason",
    "divergence_pred_50", "divergence_pred_100",
    "pred_V_dsl", "ref_V", "jacobi_ref_V", "truth_V",
    "abs_error", "pass_fail",
    "seq_length", "run_time_s", "n_layers", "d_model",
    "build_status", "run_status", "error",
]


def _setup_hull_cache():
    import runner
    from transformer_vm.attention.hull_cache import HullKVCache
    HullKVCache(1, 1)   # JIT compile prewarm
    runner.CACHE_CLASS = HullKVCache


def _decide_solver(pc, target: int) -> dict:
    """Return decision dict: solver, omega, conflicts, reason, pred_50, pred_100."""
    from coloring import two_color
    from experiments.spectrum import compute_rho_kappa
    from rbsor_reference import (
        auto_T_rbsor, detect_divergence, omega_opt,
    )

    rho, _ = compute_rho_kappa(pc)
    raw_omega = omega_opt(rho)
    omega = min(raw_omega, V01_OMEGA_CAP)
    red, black, conflicts = two_color(pc)

    if len(conflicts) > 0:
        return {
            "solver": "jacobi", "omega": omega, "rho": rho,
            "red": red, "black": black, "conflicts": conflicts,
            "reason": f"conflicts={len(conflicts)}",
            "pred_50": "", "pred_100": "",
        }

    # Bipartite: try the divergence detector at the planned omega.
    diverging, pred_a, pred_b = detect_divergence(
        pc, target, omega, red, black,
        iter_a=50, iter_b=100,
        v_step=V01_V_STEP, k_levels=V01_K_LEVELS,
    )
    if diverging:
        return {
            "solver": "jacobi", "omega": omega, "rho": rho,
            "red": red, "black": black, "conflicts": conflicts,
            "reason": f"divergence pred50={pred_a:.4f} pred100={pred_b:.4f}",
            "pred_50": f"{pred_a:.4f}", "pred_100": f"{pred_b:.4f}",
        }

    return {
        "solver": "rbsor", "omega": omega, "rho": rho,
        "red": red, "black": black, "conflicts": conflicts,
        "reason": "ok",
        "pred_50": f"{pred_a:.4f}", "pred_100": f"{pred_b:.4f}",
    }


def _model_path(cid: str, solver: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{cid}_{solver}_v01.bin")


def _build_if_missing(cid: str, decision: dict, T: int, force: bool) -> tuple[str, dict | None]:
    import build as build_mod
    out_path = _model_path(cid, decision["solver"])
    if force and os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(out_path):
        return "cached", None
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            kwargs = dict(
                T=T, out_path=out_path, plan_only=False,
                v_step=V01_V_STEP, k_levels=V01_K_LEVELS,
            )
            if decision["solver"] == "rbsor":
                kwargs["algorithm"] = "rbsor"
                kwargs["omega_cap"] = V01_OMEGA_CAP
            else:
                kwargs["algorithm"] = "jacobi"
            info = build_mod.build_for_circuit(cid, **kwargs)
        return "built", info
    except Exception as e:
        return f"build_error: {e}", None
    finally:
        devnull.close()


def _run_one(cid: str, c: dict, tol: float, force_rebuild: bool) -> dict:
    from jacobi_reference import _auto_T as jac_auto_T, jacobi_solve_parsed
    from parse import parse_netlist
    from rbsor_reference import auto_T_rbsor, rbsor_solve_parsed

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

        decision = _decide_solver(pc, target)
        row["rho_J"] = f"{decision['rho']:.6f}"
        row["omega_used"] = f"{decision['omega']:.4f}"
        row["n_red"] = len(decision["red"])
        row["n_black"] = len(decision["black"])
        row["conflicts"] = len(decision["conflicts"])
        row["solver_used"] = decision["solver"]
        row["fallback_reason"] = decision["reason"]
        row["divergence_pred_50"] = decision["pred_50"]
        row["divergence_pred_100"] = decision["pred_100"]

        # Choose T: max(500, 50N) for both algorithms in v01.
        T_v01 = max(V01_T_FLOOR, 50 * pc.num_nodes)
        if decision["solver"] == "rbsor":
            T_v01 = auto_T_rbsor(pc.num_nodes, decision["omega"], t_floor=V01_T_FLOOR)
        row["T_used"] = T_v01

        # Reference (matches DSL).
        try:
            if decision["solver"] == "rbsor":
                ref = rbsor_solve_parsed(
                    pc, target, T_v01, decision["omega"],
                    decision["red"], decision["black"],
                    v_step=V01_V_STEP, k_levels=V01_K_LEVELS,
                )
            else:
                ref = jacobi_solve_parsed(pc, target, T_v01)
            row["ref_V"] = f"{ref:.4f}"
        except Exception as e:
            log.warning(f"{cid}: ref failed: {e}")
        try:
            jv = jacobi_solve_parsed(pc, target, jac_auto_T(pc.num_nodes))
            row["jacobi_ref_V"] = f"{jv:.4f}"
        except Exception as e:
            log.warning(f"{cid}: jacobi_ref failed: {e}")

        # Build (if missing).
        bstatus, binfo = _build_if_missing(cid, decision, T_v01, force_rebuild)
        row["build_status"] = bstatus
        if binfo:
            row["n_layers"] = binfo["n_layers"]
            row["d_model"] = binfo["d_model"]
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        # Run with hull cache.
        import runner
        mpath = _model_path(cid, decision["solver"])
        t0 = time.time()
        try:
            status, pred_v, truth, _e = runner.run_circuit(
                cid, MODEL_DIR, T=T_v01, tol=tol, verbose=False,
                algorithm=decision["solver"],
                v_step=V01_V_STEP, k_levels=V01_K_LEVELS,
                omega_cap=V01_OMEGA_CAP if decision["solver"] == "rbsor" else None,
                model_path_override=mpath,
            )
            elapsed = time.time() - t0
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"] = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            row["seq_length"] = 2 * pc.num_nodes * (T_v01 + 1) + 4
        except Exception as e:
            row["run_status"] = "run_error"
            row["error"] = repr(e)
            log.warning(f"{cid}: run failed: {e}\n{traceback.format_exc()}")

    except Exception as e:
        row["error"] = repr(e)
        log.error(f"{cid}: unexpected: {e}\n{traceback.format_exc()}")

    return row


def write_rows(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="E1V01 — RB-SOR with V_STEP=0.01 + omega_cap + Jacobi fallback")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    _setup_hull_cache()
    print(f"[E1V01] cache=HullKVCache  V_STEP={V01_V_STEP}  K_LEVELS={V01_K_LEVELS}  "
          f"omega_cap={V01_OMEGA_CAP}  T_floor={V01_T_FLOOR}", flush=True)

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]
    print(f"[E1V01] {len(circuits)} circuits", flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0
    rbsor_used = jacobi_used = 0

    for i, c in enumerate(circuits):
        cid = c["ID"]
        t0 = time.time()
        row = _run_one(cid, c, tol=args.tol, force_rebuild=args.force_rebuild)
        rows.append(row)
        write_rows(rows)

        if row["solver_used"] == "rbsor": rbsor_used += 1
        elif row["solver_used"] == "jacobi": jacobi_used += 1
        if row["pass_fail"] == "PASS": pass_n += 1
        elif row["pass_fail"] == "FAIL": fail_n += 1
        else: err_n += 1

        dt = time.time() - t0
        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} N={row['N']:<3} "
            f"solver={row['solver_used']:<6} reason={row['fallback_reason'][:30]:<30} "
            f"pred={row['pred_V_dsl']:>8} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>7} [{row['pass_fail']}] {dt:.1f}s",
            flush=True,
        )

    print(
        f"\n[E1V01] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"rbsor_used={rbsor_used}  jacobi_fallback={jacobi_used}  total={len(circuits)}  "
        f"out={OUT_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
