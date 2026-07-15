"""E16 — Re-run E1 at fine V_STEP.

Re-runs all 154 circuits through the DSL evaluator with V_STEP set by
either the CLI flag --v-step or auto-detected from results/e15_decision.txt.

Outputs results_main_E16.csv with the same columns as results_main.csv.
Pass criterion: abs(pred_V - truth_V) <= 0.05V.

The MILP graph structure is unchanged (same 5 layers / d_model=36); only
the vocabulary (v_k tokens) and the CircuitMachine's quantization change.
No model.bin rebuild needed — DSL runs the graph symbolically at exact
hardmax.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import logging
import os
import time
import traceback

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("E16")

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def _detect_v_step() -> float:
    """Read the recommended V_STEP from E15's decision file, else default 0.01."""
    path = os.path.join(RESULTS_DIR, "e15_decision.txt")
    if not os.path.exists(path):
        log.warning("no e15_decision.txt; defaulting to V_STEP=0.01V")
        return 0.01
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("RECOMMENDED_V_STEP="):
                    return float(line.split("=", 1)[1].strip())
    except Exception:  # noqa: BLE001
        pass
    return 0.01


def run_e16(v_step_volts: float | None = None) -> list[dict]:
    from dsl_runner import run_one_dsl
    from jacobi_reference import _auto_T, jacobi_solve_parsed
    from parse import parse_netlist
    from spectrum import compute_rho_kappa

    if v_step_volts is None:
        v_step_volts = _detect_v_step()
    v_step_scaled = int(round(v_step_volts * 10000))
    # Keep vocabulary covering 0..24V
    k_levels = max(480, int(round(24.0 / v_step_volts)))
    log.info(f"E16 running all 154 circuits at V_STEP={v_step_volts}V "
             f"(k_levels={k_levels})")

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))

    rows: list[dict] = []
    fieldnames = [
        "circuit_id", "complexity", "N", "T_used", "V_STEP_V",
        "pred_V_dsl", "truth_V", "abs_error", "pass_fail",
        "jacobi_ref_V", "rho_M", "kappa_A",
        "seq_length", "dsl_time_ms", "error",
    ]
    out_csv = os.path.join(RESULTS_DIR, "results_main_E16.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i, c in enumerate(circuits):
        cid = c["ID"]
        row = {fn: "" for fn in fieldnames}
        row["circuit_id"] = cid
        row["complexity"] = c.get("Complexity", "")
        row["truth_V"] = c["Ground_Truth_Vout"]
        row["V_STEP_V"] = v_step_volts
        try:
            pc = parse_netlist(c["Netlist"])
            target = int(c["Target_Node"])
            T = _auto_T(pc.num_nodes)
            row["N"] = pc.num_nodes
            row["T_used"] = T

            rho, kappa = compute_rho_kappa(pc)
            row["rho_M"] = f"{rho:.6f}"
            row["kappa_A"] = f"{kappa:.4e}"

            try:
                jref = jacobi_solve_parsed(pc, target, T)
                row["jacobi_ref_V"] = f"{jref:.4f}"
            except Exception:
                pass

            t0 = time.time()
            r = run_one_dsl(c["Netlist"], target, T, parsed=pc,
                            v_step=v_step_scaled, k_levels=k_levels)
            row["pred_V_dsl"] = f"{r['pred_V']:.4f}"
            row["seq_length"] = r["seq_length"]
            row["dsl_time_ms"] = f"{(time.time() - t0) * 1000:.1f}"

            abs_err = abs(r["pred_V"] - float(c["Ground_Truth_Vout"]))
            row["abs_error"] = f"{abs_err:.4f}"
            row["pass_fail"] = "PASS" if abs_err <= 0.05 else "FAIL"
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            log.error(f"{cid}: {row['error']}")
            log.debug(traceback.format_exc())
        rows.append(row)

        # Incremental checkpoint
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        if (i + 1) % 5 == 0 or (i + 1) == len(circuits):
            passed = sum(1 for r in rows if r["pass_fail"] == "PASS")
            log.info(f"[{i+1}/{len(circuits)}] {cid} running pass={passed}")

    total = len(rows)
    passed = sum(1 for r in rows if r["pass_fail"] == "PASS")
    log.info(f"E16 final: {passed}/{total} pass at V_STEP={v_step_volts}V")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--v-step", type=float, default=None,
                   help="V_STEP in volts (auto-detects from e15_decision.txt if not set)")
    args = p.parse_args()
    run_e16(v_step_volts=args.v_step)
