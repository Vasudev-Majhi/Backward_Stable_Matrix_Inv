"""Experiment B: LU on real sparse circuit Laplacian matrices.

For each of the 154 resistor circuits, extracts the Laplacian A_FF (free-node
submatrix of the conductance matrix), builds the LU transformer, runs exact
float64 inference, and records max_abs_err vs numpy reference.

This is deliberately distinct from the voltage-token inference in lu_pipeline/:
here the output is float64 (not quantized to a voltage vocabulary), so the
result tests machine precision on real sparse SPD matrices.

Output: exp_b_circuit_laplacian.csv
Columns: circuit_id, N, kappa, sparsity, max_abs_err, median_abs_err,
         build_s, infer_s, status
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

# parse.py lives at ~/craft_release/craft/, not matrix_inversion/craft/
_MAIN_CLAUDE_FILES = os.path.expanduser("~/craft_release/craft")
if os.path.isdir(_MAIN_CLAUDE_FILES) and _MAIN_CLAUDE_FILES not in sys.path:
    sys.path.append(_MAIN_CLAUDE_FILES)

from build_lu import build_for_matrix  # noqa: E402
from runner_lu import invert  # noqa: E402
from parse import parse_netlist  # noqa: E402  (from craft/)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    os.path.expanduser("~/craft_release/dataset/circuit_dataset_rv.jsonl"),
)
MODEL_DIR = "/tmp/lu_models_circuits"
OUT_CSV = os.path.join(HERE, "exp_b_circuit_laplacian.csv")

FIELDNAMES = ["circuit_id", "N", "kappa", "sparsity",
              "max_abs_err", "median_abs_err", "build_s", "infer_s", "status"]


def build_partition(pc):
    """Return (free, fixed, A_FF, A_FP) from a ParsedCircuit."""
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g
        A[b, b] += g
        A[a, b] -= g
        A[b, a] -= g
    free = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]
    if not free:
        return free, fixed, np.zeros((0, 0), dtype=np.float64), np.zeros((0, len(fixed)))
    fi = np.array(free)
    pi = np.array(fixed)
    A_FF = A[np.ix_(fi, fi)]
    A_FP = A[np.ix_(fi, pi)] if len(fixed) > 0 else np.zeros((len(free), 0))
    return free, fixed, A_FF, A_FP


def matrix_sparsity(A: np.ndarray) -> float:
    """Fraction of near-zero entries (|a_ij| < 1e-12)."""
    n = A.size
    nnz = np.sum(np.abs(A) > 1e-12)
    return float(n - nnz) / n if n > 0 else 0.0


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    circuits = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line.strip()))
    print(f"Loaded {len(circuits)} circuits from {DATASET_PATH}", flush=True)

    csv_f = open(OUT_CSV, "w", newline="", buffering=1)
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    writer.writeheader()
    csv_f.flush()

    print(f"{'cid':>12} {'N':>4} {'kappa':>10} {'sparsity':>9} "
          f"{'max_err':>12} {'build_s':>8} {'infer_s':>8}", flush=True)

    for c in circuits:
        cid = c["ID"]
        try:
            pc = parse_netlist(c["Netlist"])
            free, fixed, A_FF, _ = build_partition(pc)
            n = len(free)

            if n == 0:
                row = {"circuit_id": cid, "N": 0, "status": "skip_no_free",
                       "kappa": "", "sparsity": "", "max_abs_err": "",
                       "median_abs_err": "", "build_s": "", "infer_s": ""}
                writer.writerow(row)
                csv_f.flush()
                continue

            kappa = float(np.linalg.cond(A_FF))
            sparsity = matrix_sparsity(A_FF)

            t0 = time.time()
            r = build_for_matrix(A_FF, model_dir=MODEL_DIR)
            build_s = time.time() - t0

            t0 = time.time()
            X = invert(A_FF, r["model_path"])
            infer_s = time.time() - t0

            Xref = np.linalg.inv(A_FF)
            errs = np.abs(X - Xref)
            max_err = float(errs.max())
            med_err = float(np.median(errs))

            row = {
                "circuit_id": cid, "N": n,
                "kappa": f"{kappa:.4e}", "sparsity": f"{sparsity:.4f}",
                "max_abs_err": max_err, "median_abs_err": med_err,
                "build_s": round(build_s, 3), "infer_s": round(infer_s, 3),
                "status": "ok",
            }
            writer.writerow(row)
            csv_f.flush()
            print(f"{cid:>12} {n:>4} {kappa:>10.2e} {sparsity:>9.4f} "
                  f"{max_err:>12.3e} {build_s:>8.2f} {infer_s:>8.2f}", flush=True)

        except Exception as exc:
            row = {"circuit_id": cid, "N": "", "kappa": "", "sparsity": "",
                   "max_abs_err": "", "median_abs_err": "", "build_s": "",
                   "infer_s": "", "status": f"error:{exc}"}
            writer.writerow(row)
            csv_f.flush()
            print(f"{cid:>12}  ERROR: {exc}", flush=True)

    csv_f.close()
    print(f"\nDone. Results saved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
