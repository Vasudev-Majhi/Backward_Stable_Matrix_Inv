"""E18 — omega sweep on the 14 regression circuits at V_STEP=0.05.

For each of the 14 RB-SOR regressions identified in E1, run the RB-SOR DSL
at omega in {1.0, 1.3, 1.5, 1.7, auto=omega_opt(rho)}. Goal: produce a clean
plot of error vs omega showing the SOR instability boundary.

Output:
  rbsor_results/omega_sweep_E18.csv   (per (cid, omega) row)
  rbsor_results/figures/omega_instability.png   (one panel per circuit)

Hull cache enabled module-globally.
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
OUT_CSV = os.path.join(RESULTS_DIR, "omega_sweep_E18.csv")
FIG_PATH = os.path.join(RESULTS_DIR, "figures", "omega_instability.png")

REGRESSIONS = [
    "CKT_0059", "CKT_0066", "CKT_0080", "CKT_0081", "CKT_0103",
    "CKT_0113", "CKT_0116", "CKT_0120", "CKT_0123", "CKT_0128",
    "CKT_0140", "CKT_0141", "CKT_0145", "CKT_0148",
]

OMEGA_GRID_BASE = [1.0, 1.3, 1.5, 1.7]   # plus "auto" per circuit

FIELDS = [
    "circuit_id", "N", "rho_J", "omega", "omega_label",
    "T_used", "pred_V_dsl", "ref_V_rbsor", "truth_V",
    "abs_error", "pass_fail", "run_time_s",
    "build_status", "run_status", "error",
]


def _setup_hull():
    import runner
    from transformer_vm.attention.hull_cache import HullKVCache
    HullKVCache(1, 1)
    runner.CACHE_CLASS = HullKVCache


def _model_path(cid: str, omega: float) -> str:
    # Encode omega as 3-digit fixed (1.0 -> 100, 1.93 -> 193).
    tag = f"{int(round(omega * 100)):03d}"
    return os.path.join(MODEL_DIR, f"model_{cid}_rbsor_om{tag}.bin")


def _build_omega(cid: str, omega: float, T: int) -> tuple[str, dict | None]:
    import build as build_mod
    out = _model_path(cid, omega)
    if os.path.exists(out):
        return "cached", None
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            info = build_mod.build_for_circuit(
                cid, T=T, out_path=out, plan_only=False,
                algorithm="rbsor", omega_cap=omega,
            )
        return "built", info
    except Exception as e:
        return f"build_error: {e}", None
    finally:
        devnull.close()


def _run_one(cid: str, c: dict, omega: float, omega_label: str, tol: float) -> dict:
    from coloring import two_color
    from experiments.spectrum import compute_rho_kappa
    from parse import parse_netlist
    from rbsor_reference import auto_T_rbsor, rbsor_solve_parsed

    row = {fn: "" for fn in FIELDS}
    row["circuit_id"] = cid
    row["omega"] = f"{omega:.4f}"
    row["omega_label"] = omega_label
    row["truth_V"] = c["Ground_Truth_Vout"]

    try:
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        pc = parse_netlist(netlist)
        row["N"] = pc.num_nodes
        rho, _ = compute_rho_kappa(pc)
        row["rho_J"] = f"{rho:.6f}"
        red, black, _ = two_color(pc)
        T = auto_T_rbsor(pc.num_nodes, omega)
        row["T_used"] = T

        try:
            ref = rbsor_solve_parsed(pc, target, T, omega, red, black)
            row["ref_V_rbsor"] = f"{ref:.4f}"
        except Exception as e:
            log.warning(f"{cid} om={omega}: ref failed {e}")

        bstatus, _ = _build_omega(cid, omega, T)
        row["build_status"] = bstatus
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        import runner
        mpath = _model_path(cid, omega)
        t0 = time.time()
        status, pred, truth, _ = runner.run_circuit(
            cid, MODEL_DIR, T=T, tol=tol, verbose=False,
            algorithm="rbsor", omega_cap=omega,
            model_path_override=mpath,
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

    by_cid: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        try:
            cid = r["circuit_id"]
            om = float(r["omega"])
            err = float(r["abs_error"])
            by_cid.setdefault(cid, []).append((om, err))
        except (ValueError, KeyError):
            continue

    n = len(by_cid)
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)
    for ax, (cid, pts) in zip(axes.flat, sorted(by_cid.items())):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o")
        ax.axhline(0.05, ls="--", c="g", label="tol 0.05V")
        ax.set_xlabel("omega")
        ax.set_ylabel("|err| V")
        ax.set_title(cid)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    # Blank unused subplots.
    for ax in axes.flat[len(by_cid):]:
        ax.axis("off")

    fig.suptitle("E18 — RB-SOR error vs omega (14 regression circuits, V_STEP=0.05)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120)
    print(f"[E18] wrote {FIG_PATH}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E18 — omega sweep on regression circuits")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    _setup_hull()
    print(f"[E18] cache=HullKVCache  circuits={len(REGRESSIONS)}  omegas={OMEGA_GRID_BASE}+auto", flush=True)

    by_id = {}
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            by_id[c["ID"]] = c

    from coloring import two_color
    from experiments.spectrum import compute_rho_kappa
    from parse import parse_netlist
    from rbsor_reference import omega_opt

    rows: list[dict] = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for cid in REGRESSIONS:
        if cid not in by_id:
            print(f"[E18] WARN missing {cid}", flush=True)
            continue
        c = by_id[cid]
        # Compute auto-omega for this circuit (uncapped).
        try:
            pc = parse_netlist(c["Netlist"])
            rho, _ = compute_rho_kappa(pc)
            auto_om = omega_opt(rho)
        except Exception as e:
            print(f"[E18] {cid} compute_rho failed: {e}", flush=True)
            continue
        omegas: list[tuple[float, str]] = [(om, f"{om:.2f}") for om in OMEGA_GRID_BASE]
        omegas.append((auto_om, "auto"))

        for omega, label in omegas:
            row = _run_one(cid, c, omega, label, args.tol)
            rows.append(row)
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(
                f"[E18] {cid} om={omega:.4f} ({label:>4}) "
                f"pred={row['pred_V_dsl']:>8} err={row['abs_error']:>7} "
                f"[{row['pass_fail']}] {row['run_time_s']}s",
                flush=True,
            )

    _make_figure(rows)
    print(f"\n[E18] DONE  rows={len(rows)}  out={OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
