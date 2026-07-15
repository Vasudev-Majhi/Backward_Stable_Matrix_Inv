"""E20 — Hull benchmark on heat 10x10 (N=101, T~5050).

Validates Hull cache scaling at N>50 by running the 10x10 heat-grid problem
with both StandardKVCache and HullKVCache. Reports wall-clock per cache.

Note: this uses the EXISTING Jacobi pipeline (heat is the canonical Jacobi
test); no RB-SOR code involved. Output augments heat_diffusion CSV with
runtime_standard_s and runtime_hull_s columns.

Output:
  rbsor_results/heat_hull_E20.csv
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
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

RESULTS_DIR = os.path.join(REPO_ROOT, "rbsor_results")
OUT_CSV = os.path.join(RESULTS_DIR, "heat_hull_E20.csv")

GRIDS = [(10, 10)]   # only N=101 needed; smaller grids already covered by E11

FIELDS = [
    "grid", "rows", "cols", "N", "seq_length", "T",
    "pred_V_standard", "pred_V_hull",
    "runtime_standard_s", "runtime_hull_s", "speedup",
    "predictions_match", "error",
]


def _build_heat_model(rows: int, cols: int) -> tuple[str, str, int, float]:
    """Build a heat-grid Jacobi model and return (cid, model_path, target, analytical)."""
    from heat_netlist import thermal_grid_to_spice
    from parse import parse_netlist
    import build as build_mod

    # Asymmetric BCs to avoid the half-step degenerate case (per heat_netlist comments).
    netlist, target, analytical = thermal_grid_to_spice(
        rows, cols, top_temp=10.0, bottom_temp=0.0, left_temp=4.0, right_temp=7.0,
    )
    cid = f"heat_{rows}x{cols}"

    # Persist the netlist into a tempfile-style "synthetic dataset" so build/runner
    # can find it via load_circuit. Simpler: monkey-patch by writing a single-line
    # JSONL alongside and overriding CRAFT_DATASET temporarily.
    pc = parse_netlist(netlist)
    N = pc.num_nodes

    # Stash a synthetic dataset in a temp file; build_for_circuit uses
    # CRAFT_DATASET implicitly via load_circuit.
    tmp_jsonl = os.path.join(tempfile.gettempdir(), f"_heat_{rows}x{cols}.jsonl")
    with open(tmp_jsonl, "w") as f:
        f.write(json.dumps({
            "ID": cid,
            "Complexity": "Heat",
            "Netlist": netlist,
            "Target_Node": target,
            "Ground_Truth_Vout": analytical,
        }) + "\n")

    out_path = os.path.join(REPO_ROOT, f"model_{cid}.bin")
    if os.path.exists(out_path):
        return cid, out_path, target, analytical

    # T = max(1000, 50N) is the standard Jacobi formula (= 5050 for N=101).
    from jacobi_reference import _auto_T
    T = _auto_T(N)

    # build_for_circuit reads from build_mod.DATASET_PATH; override.
    saved = build_mod.DATASET_PATH
    build_mod.DATASET_PATH = tmp_jsonl
    try:
        devnull = open(os.devnull, "w")
        try:
            with redirect_stdout(devnull):
                build_mod.build_for_circuit(
                    cid, T=T, out_path=out_path, plan_only=False, algorithm="jacobi",
                )
        finally:
            devnull.close()
    finally:
        build_mod.DATASET_PATH = saved

    return cid, out_path, target, analytical


def _run_with_cache(cid: str, mpath: str, target: int, analytical: float,
                    cache_name: str, T: int, netlist: str) -> tuple[float, float]:
    """Run inference with the named cache. Returns (pred_V, elapsed_s)."""
    import runner

    if cache_name == "hull":
        from transformer_vm.attention.hull_cache import HullKVCache
        runner.CACHE_CLASS = HullKVCache
    elif cache_name == "standard":
        from transformer_vm.attention.standard_cache import StandardKVCache
        runner.CACHE_CLASS = StandardKVCache
    else:
        raise ValueError(cache_name)

    # Inject a temp dataset entry so run_circuit's load_dataset finds it.
    tmp_jsonl = os.path.join(tempfile.gettempdir(), f"_heat_{cid}.jsonl")
    with open(tmp_jsonl, "w") as f:
        f.write(json.dumps({
            "ID": cid, "Complexity": "Heat",
            "Netlist": netlist, "Target_Node": target,
            "Ground_Truth_Vout": analytical,
        }) + "\n")

    saved_ds = runner.DATASET_PATH
    runner.DATASET_PATH = tmp_jsonl
    try:
        t0 = time.time()
        status, pred, _, _ = runner.run_circuit(
            cid, REPO_ROOT, T=T, tol=0.1, verbose=False,
            algorithm="jacobi", model_path_override=mpath,
        )
        elapsed = time.time() - t0
    finally:
        runner.DATASET_PATH = saved_ds

    return pred, elapsed


def main():
    ap = argparse.ArgumentParser(description="E20 — heat 10x10 hull vs standard")
    ap.add_argument("--skip-standard", action="store_true",
                    help="Skip Standard cache run (e.g. if too slow)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows_out: list[dict] = []

    for (r, c) in GRIDS:
        from heat_netlist import thermal_grid_to_spice
        from jacobi_reference import _auto_T
        from parse import parse_netlist
        netlist, target, analytical = thermal_grid_to_spice(
            r, c, top_temp=10.0, bottom_temp=0.0, left_temp=4.0, right_temp=7.0,
        )
        pc = parse_netlist(netlist)
        N = pc.num_nodes
        T = _auto_T(N)
        seq_len = 2 * N * (T + 1) + 4

        print(f"[E20] grid={r}x{c}  N={N}  T={T}  seq_len={seq_len}", flush=True)

        cid, mpath, target, analytical = _build_heat_model(r, c)
        print(f"[E20] built {mpath}  analytical={analytical:.4f}V", flush=True)

        row = {fn: "" for fn in FIELDS}
        row["grid"] = f"{r}x{c}"
        row["rows"] = r
        row["cols"] = c
        row["N"] = N
        row["T"] = T
        row["seq_length"] = seq_len

        # Hull first (fast), then standard.
        try:
            pred_h, t_h = _run_with_cache(cid, mpath, target, analytical, "hull", T, netlist)
            row["pred_V_hull"] = f"{pred_h:.4f}"
            row["runtime_hull_s"] = f"{t_h:.2f}"
            print(f"[E20] hull:    pred={pred_h:.4f}V  time={t_h:.1f}s", flush=True)
        except Exception as e:
            row["error"] = f"hull: {e}"
            print(f"[E20] hull FAILED: {e}", flush=True)

        if not args.skip_standard:
            try:
                pred_s, t_s = _run_with_cache(cid, mpath, target, analytical, "standard", T, netlist)
                row["pred_V_standard"] = f"{pred_s:.4f}"
                row["runtime_standard_s"] = f"{t_s:.2f}"
                print(f"[E20] standard: pred={pred_s:.4f}V  time={t_s:.1f}s", flush=True)
            except Exception as e:
                row["error"] = (row["error"] + " | " if row["error"] else "") + f"standard: {e}"
                print(f"[E20] standard FAILED: {e}", flush=True)

            # Speedup + match.
            try:
                t_h = float(row["runtime_hull_s"])
                t_s = float(row["runtime_standard_s"])
                row["speedup"] = f"{t_s / t_h:.2f}"
                row["predictions_match"] = "Y" if abs(
                    float(row["pred_V_hull"]) - float(row["pred_V_standard"])
                ) < 1e-4 else "N"
            except (ValueError, KeyError):
                pass

        rows_out.append(row)

        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)

    print(f"\n[E20] DONE  out={OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
