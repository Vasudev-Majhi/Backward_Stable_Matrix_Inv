"""E19 — RB-SOR T-scaling on 9 rescues + 14 regressions (23 circuits).

For each circuit, run RB-SOR DSL at T in {50, 100, 200, 500, 1000, 2000}
with the per-circuit auto-omega (uncapped). Plot error vs T. This separates:
  - Borderline regressions  (error decreases with T -> just needs more iters)
  - Catastrophic divergences (error grows with T -> truly diverging)

Output:
  rbsor_results/rbsor_tscaling_E19.csv
  rbsor_results/figures/rbsor_tscaling.png
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
OUT_CSV = os.path.join(RESULTS_DIR, "rbsor_tscaling_E19.csv")
FIG_PATH = os.path.join(RESULTS_DIR, "figures", "rbsor_tscaling.png")

RESCUES = [
    "CKT_0015", "CKT_0032", "CKT_0035", "CKT_0048", "CKT_0069",
    "CKT_0089", "CKT_0105", "CKT_0124", "CKT_0137",
]
REGRESSIONS = [
    "CKT_0059", "CKT_0066", "CKT_0080", "CKT_0081", "CKT_0103",
    "CKT_0113", "CKT_0116", "CKT_0120", "CKT_0123", "CKT_0128",
    "CKT_0140", "CKT_0141", "CKT_0145", "CKT_0148",
]
ALL_23 = RESCUES + REGRESSIONS

T_GRID = [50, 100, 200, 500, 1000, 2000]

FIELDS = [
    "circuit_id", "category", "N", "rho_J", "omega",
    "T", "pred_V_dsl", "ref_V_rbsor", "truth_V",
    "abs_error", "pass_fail", "run_time_s",
    "build_status", "run_status", "error",
]


def _setup_hull():
    import runner
    from transformer_vm.attention.hull_cache import HullKVCache
    HullKVCache(1, 1)
    runner.CACHE_CLASS = HullKVCache


def _model_path(cid: str, T: int) -> str:
    return os.path.join(MODEL_DIR, f"model_{cid}_rbsor_T{T:05d}.bin")


def _build_for_T(cid: str, T: int) -> tuple[str, dict | None]:
    import build as build_mod
    out = _model_path(cid, T)
    if os.path.exists(out):
        return "cached", None
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            info = build_mod.build_for_circuit(
                cid, T=T, out_path=out, plan_only=False, algorithm="rbsor",
            )
        return "built", info
    except Exception as e:
        return f"build_error: {e}", None
    finally:
        devnull.close()


def _run_one(cid: str, c: dict, T: int, category: str, tol: float) -> dict:
    from coloring import two_color
    from experiments.spectrum import compute_rho_kappa
    from parse import parse_netlist
    from rbsor_reference import omega_opt, rbsor_solve_parsed

    row = {fn: "" for fn in FIELDS}
    row["circuit_id"] = cid
    row["category"] = category
    row["T"] = T
    row["truth_V"] = c["Ground_Truth_Vout"]

    try:
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        pc = parse_netlist(netlist)
        row["N"] = pc.num_nodes
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        row["rho_J"] = f"{rho:.6f}"
        row["omega"] = f"{omega:.4f}"
        red, black, _ = two_color(pc)

        try:
            ref = rbsor_solve_parsed(pc, target, T, omega, red, black)
            row["ref_V_rbsor"] = f"{ref:.4f}"
        except Exception as e:
            log.warning(f"{cid} T={T}: ref failed {e}")

        bstatus, _ = _build_for_T(cid, T)
        row["build_status"] = bstatus
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        import runner
        mpath = _model_path(cid, T)
        t0 = time.time()
        status, pred, truth, _ = runner.run_circuit(
            cid, MODEL_DIR, T=T, tol=tol, verbose=False,
            algorithm="rbsor", model_path_override=mpath,
        )
        elapsed = time.time() - t0
        row["pred_V_dsl"] = f"{pred:.4f}"
        row["abs_error"] = f"{abs(pred - truth):.4f}"
        row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
        row["run_status"] = status
        row["run_time_s"] = f"{elapsed:.2f}"
    except Exception as e:
        row["run_status"] = "run_error"
        row["error"] = repr(e)

    return row


def _make_figure(rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping figure")
        return

    by_cid: dict[str, dict] = {}
    for r in rows:
        try:
            cid = r["circuit_id"]
            T = int(r["T"])
            err = float(r["abs_error"])
            cat = r["category"]
            by_cid.setdefault(cid, {"cat": cat, "pts": []})["pts"].append((T, err))
        except (ValueError, KeyError):
            continue

    n = len(by_cid)
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)
    for ax, (cid, info) in zip(axes.flat, sorted(by_cid.items())):
        info["pts"].sort()
        xs = [p[0] for p in info["pts"]]
        ys = [p[1] for p in info["pts"]]
        color = "g" if info["cat"] == "rescue" else "r"
        ax.plot(xs, ys, marker="o", color=color)
        ax.axhline(0.05, ls="--", c="grey", label="tol")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("T")
        ax.set_ylabel("|err| V")
        ax.set_title(f"{cid} ({info['cat']})")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(by_cid):]:
        ax.axis("off")

    fig.suptitle("E19 — RB-SOR error vs T (green=rescue, red=regression)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120)
    print(f"[E19] wrote {FIG_PATH}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E19 — RB-SOR T scaling on 23 circuits")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    _setup_hull()
    print(f"[E19] cache=HullKVCache  circuits={len(ALL_23)}  T={T_GRID}", flush=True)

    by_id = {}
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            by_id[c["ID"]] = c

    rows: list[dict] = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for cid in ALL_23:
        if cid not in by_id:
            print(f"[E19] WARN missing {cid}", flush=True)
            continue
        category = "rescue" if cid in RESCUES else "regression"
        for T in T_GRID:
            row = _run_one(cid, by_id[cid], T, category, args.tol)
            rows.append(row)
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(
                f"[E19] {cid} ({category[:3]:<3}) T={T:>5} "
                f"pred={row['pred_V_dsl']:>8} err={row['abs_error']:>7} "
                f"[{row['pass_fail']}] {row['run_time_s']}s",
                flush=True,
            )

    _make_figure(rows)
    print(f"\n[E19] DONE  rows={len(rows)}  out={OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
