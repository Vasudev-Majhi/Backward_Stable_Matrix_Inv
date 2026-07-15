"""R1 — Forward RB-SOR full-dataset sweep with HullKVCache.

Mirrors E1 (the Jacobi 154-circuit sweep) for the RB-SOR algorithm.
Designed for autonomous execution under tmux: incremental CSV checkpoints
after every circuit so partial results survive crashes / kills.

For each circuit:
  1. Build model_<ID>_rbsor.bin if missing (per-circuit weights)
  2. Run runner.run_circuit with HullKVCache injected and --algorithm rbsor
  3. Compute reference prediction (Python rbsor solver, mirrors DSL semantics)
  4. Append a row to results/results_main_rbsor.csv

Output schema (key columns):
  circuit_id, complexity, N, max_degree, target_node,
  rho_J, omega, T_used, n_red, n_black, conflicts,
  pred_V_dsl, ref_V_rbsor, jacobi_ref_V, truth_V,
  abs_error, pass_fail, dsl_matches_ref,
  seq_length, run_time_s, n_layers, d_model,
  build_status, run_status, error

Usage:
  python rbsor_run_all.py                  # full 154
  python rbsor_run_all.py --limit 10       # first 10 (sanity)
  python rbsor_run_all.py --skip-build     # assume models already built
  python rbsor_run_all.py --no-hull        # use StandardKVCache (slow)
  python rbsor_run_all.py --tol 0.05       # pass tolerance in volts
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

# Add rbsor/ folder to sys.path so we can import the new modules.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RBSOR_DIR = os.path.join(REPO_ROOT, "rbsor")
if RBSOR_DIR not in sys.path:
    sys.path.insert(0, RBSOR_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
MODEL_DIR = REPO_ROOT
# All RB-SOR outputs land in craft/rbsor_results/ (sibling of experiments/).
RESULTS_DIR = os.path.join(REPO_ROOT, "rbsor_results")
OUT_CSV = os.path.join(RESULTS_DIR, "results_main_rbsor.csv")

FIELDS = [
    "circuit_id", "complexity", "N", "max_degree", "target_node",
    "rho_J", "omega", "T_used", "n_red", "n_black", "conflicts",
    "pred_V_dsl", "ref_V_rbsor", "jacobi_ref_V", "truth_V",
    "abs_error", "pass_fail", "dsl_matches_ref",
    "seq_length", "run_time_s", "n_layers", "d_model",
    "build_status", "run_status", "error",
]


def _setup_hull_cache(use_hull: bool) -> str:
    """Inject the cache class into runner.CACHE_CLASS. Returns the class name."""
    import runner
    if use_hull:
        from transformer_vm.attention.hull_cache import HullKVCache
        # Pre-warm JIT compile so it doesn't happen mid-loop on first circuit.
        HullKVCache(1, 1)
        runner.CACHE_CLASS = HullKVCache
        return "HullKVCache"
    else:
        from transformer_vm.attention.standard_cache import StandardKVCache
        runner.CACHE_CLASS = StandardKVCache
        return "StandardKVCache"


def _build_model_if_missing(cid: str, model_path: str) -> tuple[str, dict | None]:
    """Build model_<cid>_rbsor.bin if it doesn't exist. Returns (status, build_info)."""
    if os.path.exists(model_path):
        return "cached", None
    import build as build_mod
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            info = build_mod.build_for_circuit(
                cid, T=None, out_path=model_path, plan_only=False,
                algorithm="rbsor",
            )
        return "built", info
    except Exception as e:
        return f"build_error: {e}", None
    finally:
        devnull.close()


def _run_one_circuit(
    cid: str, c: dict, tol: float, force_rebuild: bool,
) -> dict:
    """Build (if missing) + run one circuit. Returns CSV row dict."""
    from coloring import two_color
    from experiments.spectrum import compute_rho_kappa
    from jacobi_reference import _auto_T as jac_auto_T, jacobi_solve_parsed
    from parse import parse_netlist
    from rbsor_reference import auto_T_rbsor, omega_opt, rbsor_solve_parsed

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

        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        T_rb = auto_T_rbsor(pc.num_nodes, omega)
        red, black, conflicts = two_color(pc)
        row["rho_J"] = f"{rho:.6f}"
        row["omega"] = f"{omega:.4f}"
        row["T_used"] = T_rb
        row["n_red"] = len(red)
        row["n_black"] = len(black)
        row["conflicts"] = len(conflicts)

        # Reference (RB-SOR + Jacobi) — both fast in pure Python.
        try:
            ref_v = rbsor_solve_parsed(pc, target, T_rb, omega, red, black)
            row["ref_V_rbsor"] = f"{ref_v:.4f}"
        except Exception as e:
            row["ref_V_rbsor"] = ""
            log.warning(f"{cid}: rbsor_ref failed: {e}")
        try:
            jac_v = jacobi_solve_parsed(pc, target, jac_auto_T(pc.num_nodes))
            row["jacobi_ref_V"] = f"{jac_v:.4f}"
        except Exception as e:
            row["jacobi_ref_V"] = ""
            log.warning(f"{cid}: jacobi_ref failed: {e}")

        # Build (if missing).
        model_path = os.path.join(MODEL_DIR, f"model_{cid}_rbsor.bin")
        if force_rebuild and os.path.exists(model_path):
            os.remove(model_path)
        bstatus, binfo = _build_model_if_missing(cid, model_path)
        row["build_status"] = bstatus
        if binfo:
            row["n_layers"] = binfo["n_layers"]
            row["d_model"] = binfo["d_model"]
        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row
        if bstatus == "cached":
            # Reload to pick up shape info for the CSV.
            try:
                from transformer_vm.model.weights import load_weights
                m, _, _ = load_weights(model_path)
                row["n_layers"] = len(m.attn)
                row["d_model"] = m.tok.weight.shape[1]
            except Exception:
                pass

        # Run with the configured cache.
        import runner
        t0 = time.time()
        try:
            status, pred_v, truth, _elapsed = runner.run_circuit(
                cid, MODEL_DIR, T=None, tol=tol, verbose=False,
                algorithm="rbsor",
            )
            elapsed = time.time() - t0
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"] = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            # 2N(T+1) + 4 — same formula as the existing Jacobi pipeline.
            row["seq_length"] = 2 * pc.num_nodes * (T_rb + 1) + 4
            # DSL ↔ reference parity flag.
            try:
                ref_f = float(row["ref_V_rbsor"]) if row["ref_V_rbsor"] else None
                row["dsl_matches_ref"] = (
                    "Y" if (ref_f is not None and abs(pred_v - ref_f) < 0.01) else "N"
                )
            except Exception:
                row["dsl_matches_ref"] = ""
        except Exception as e:
            row["run_status"] = f"run_error"
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
    ap = argparse.ArgumentParser(description="R1 — RB-SOR full-dataset sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-build", action="store_true",
                    help="Don't build new models; only run those already on disk")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Delete and rebuild any existing model_<ID>_rbsor.bin")
    ap.add_argument("--no-hull", action="store_true",
                    help="Use StandardKVCache (much slower; for debugging only)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="Pass tolerance in volts (default 0.05)")
    args = ap.parse_args()

    # Cache injection.
    cache_name = _setup_hull_cache(use_hull=not args.no_hull)
    print(f"[rbsor_run_all] cache = {cache_name}", flush=True)

    # Dataset.
    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]
    print(f"[rbsor_run_all] {len(circuits)} circuits", flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0
    rescue_n = 0   # Jacobi-FAIL but RB-SOR-PASS

    for i, c in enumerate(circuits):
        cid = c["ID"]
        t0 = time.time()
        row = _run_one_circuit(
            cid, c, tol=args.tol, force_rebuild=args.force_rebuild,
        )
        rows.append(row)
        write_rows(rows)   # incremental checkpoint

        # Console summary line.
        bucket = "?"
        if row["pass_fail"] == "PASS":
            pass_n += 1
            try:
                jv = float(row["jacobi_ref_V"]) if row["jacobi_ref_V"] else None
                tv = float(row["truth_V"])
                jacobi_passed = jv is not None and abs(jv - tv) <= args.tol
                if not jacobi_passed:
                    rescue_n += 1
                    bucket = "RESCUE"
                else:
                    bucket = "PASS"
            except Exception:
                bucket = "PASS"
        elif row["pass_fail"] == "FAIL":
            fail_n += 1
            bucket = "FAIL"
        else:
            err_n += 1
            bucket = row["run_status"] or "ERR"

        dt = time.time() - t0
        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} "
            f"N={row['N']:<3} rho={row['rho_J']} om={row['omega']} "
            f"T={row['T_used']:<5} pred={row['pred_V_dsl']:>8} "
            f"truth={row['truth_V']:>8.4f} err={row['abs_error']:>7} "
            f"[{bucket}] {dt:.1f}s",
            flush=True,
        )

    # Final summary line so the wrapper script can grep for it.
    print(
        f"\n[rbsor_run_all] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"rescues_vs_jacobi={rescue_n}  total={len(circuits)}  "
        f"out={OUT_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
