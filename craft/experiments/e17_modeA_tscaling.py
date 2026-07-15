"""E17 — Mode A T-scaling.

Confirms that for high-ρ (Mode A) circuits, BOTH the DSL and the Python
Jacobi reference stay far from truth at every T — meaning the failure is
algorithmic (Jacobi itself can't solve the circuit), not a DSL artefact.

Takes CKT_0133 (ρ=1.0000, 7x7 grid) by default. Runs at
T ∈ {1, 10, 100, 1000, 5000}. Records DSL pred, Jacobi ref, and truth.
Output: mode_a_tscaling_E17.csv.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("E17")

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODE_A_CIRCUITS = ["CKT_0133"]   # 7x7 grid mesh, rho ~ 1.0
T_VALUES = [1, 10, 100, 1000, 5000]


def run_e17(circuit_ids: list[str] | None = None) -> list[dict]:
    from dsl_runner import run_one_dsl
    from jacobi_reference import jacobi_solve_parsed
    from parse import parse_netlist
    from spectrum import compute_rho_kappa

    ids = circuit_ids if circuit_ids is not None else MODE_A_CIRCUITS

    circuits: dict[str, dict] = {}
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] in ids:
                circuits[c["ID"]] = c

    rows: list[dict] = []
    for cid in ids:
        if cid not in circuits:
            log.warning(f"{cid}: not in dataset")
            continue
        c = circuits[cid]
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = float(c["Ground_Truth_Vout"])
        rho, kappa = compute_rho_kappa(pc)
        log.info(f"{cid}: N={pc.num_nodes} rho={rho:.6f} truth={truth:.4f}")

        for T in T_VALUES:
            t0 = time.time()
            try:
                # Jacobi reference first (fast)
                jref = jacobi_solve_parsed(pc, target, T)
                # DSL prediction
                r = run_one_dsl(c["Netlist"], target, T, parsed=pc)
                err_jacobi_vs_truth = abs(jref - truth)
                err_dsl_vs_truth = abs(r["pred_V"] - truth)
                err_dsl_vs_jacobi = abs(r["pred_V"] - jref)
                row = {
                    "circuit_id": cid,
                    "N": pc.num_nodes,
                    "rho_M": f"{rho:.6f}",
                    "kappa_A": f"{kappa:.4e}",
                    "T": T,
                    "truth_V": f"{truth:.4f}",
                    "jacobi_ref_V": f"{jref:.6f}",
                    "dsl_pred_V": f"{r['pred_V']:.4f}",
                    "err_jacobi_vs_truth": f"{err_jacobi_vs_truth:.4f}",
                    "err_dsl_vs_truth": f"{err_dsl_vs_truth:.4f}",
                    "err_dsl_vs_jacobi": f"{err_dsl_vs_jacobi:.4f}",
                    "seq_length": r["seq_length"],
                    "runtime_s": f"{time.time() - t0:.2f}",
                    "error": "",
                }
                log.info(
                    f"  T={T:>5}: jref={jref:.4f} dsl={r['pred_V']:.4f} "
                    f"truth={truth:.4f} err_jacobi={err_jacobi_vs_truth:.4f} "
                    f"err_dsl={err_dsl_vs_truth:.4f}"
                )
            except Exception as e:  # noqa: BLE001
                log.error(f"  T={T}: {e}")
                row = {"circuit_id": cid, "T": T,
                       "error": f"{type(e).__name__}: {e}"}
            rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "mode_a_tscaling_E17.csv")
    fieldnames = [
        "circuit_id", "N", "rho_M", "kappa_A", "T",
        "truth_V", "jacobi_ref_V", "dsl_pred_V",
        "err_jacobi_vs_truth", "err_dsl_vs_truth", "err_dsl_vs_jacobi",
        "seq_length", "runtime_s", "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--circuits", default=",".join(MODE_A_CIRCUITS),
                   help="comma-separated circuit IDs")
    args = p.parse_args()
    run_e17(circuit_ids=args.circuits.split(","))
