"""E13 — T scaling with theoretical ρ^T overlay.

Runs 6 circuits spanning ρ ∈ [0, 0.999] at 12 log-spaced T values.
Generates data for a plot that overlays the empirical abs_error(T)
against the theoretical curve err(T) ≈ err(T=1) · ρ^(T-1).

If the empirical and theoretical curves coincide on a log-log plot, this is
direct experimental proof of the spectral-radius convergence theorem for
Jacobi iteration, and justifies the failure-boundary in the phase diagram.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("E13")

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Circuits chosen by approximate ρ.
E13_CIRCUITS = [
    "CKT_0001",   # rho ≈ 0  (trivial)
    "CKT_0022",   # rho ≈ 0.1
    "CKT_0058",   # rho ≈ 0.5
    "CKT_0067",   # rho ≈ 0.85
    "CKT_0146",   # rho ≈ 0.97 (Hard tier, within reasonable T budget)
    "CKT_0131",   # rho ≈ 0.999  (near limit, expected to fail)
]
T_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def run_e13() -> list[dict]:
    from dsl_runner import run_one_dsl
    from parse import parse_netlist
    from spectrum import compute_rho_kappa

    circuits: dict[str, dict] = {}
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] in E13_CIRCUITS:
                circuits[c["ID"]] = c

    rows: list[dict] = []

    for cid in E13_CIRCUITS:
        if cid not in circuits:
            log.warning(f"{cid}: not in dataset, skipping")
            continue
        c = circuits[cid]
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = float(c["Ground_Truth_Vout"])
        rho, kappa = compute_rho_kappa(pc)
        log.info(f"{cid}: N={pc.num_nodes} rho={rho:.4f}")

        for T in T_VALUES:
            t0 = time.time()
            try:
                r = run_one_dsl(c["Netlist"], target, T, parsed=pc)
                err = abs(r["pred_V"] - truth)
                row = {
                    "circuit_id": cid,
                    "N": pc.num_nodes,
                    "rho_M": f"{rho:.6f}",
                    "T": T,
                    "pred_V": f"{r['pred_V']:.4f}",
                    "truth_V": f"{truth:.4f}",
                    "abs_error": f"{err:.6f}",
                    "seq_length": r["seq_length"],
                    "runtime_s": f"{time.time() - t0:.2f}",
                    "error": "",
                }
                log.info(f"  T={T:>5}: pred={r['pred_V']:.4f} err={err:.4f}")
            except Exception as e:  # noqa: BLE001
                log.error(f"  T={T}: {e}")
                row = {
                    "circuit_id": cid, "rho_M": f"{rho:.6f}", "T": T,
                    "error": f"{type(e).__name__}: {e}",
                }
            rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "tscaling_E13.csv")
    fieldnames = [
        "circuit_id", "N", "rho_M", "T",
        "pred_V", "truth_V", "abs_error",
        "seq_length", "runtime_s", "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    run_e13()
