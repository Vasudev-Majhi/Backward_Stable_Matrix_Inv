"""Compute condition number kappa(A_FF) for all 154 circuits.

Outputs: ~/craft_release/craft/cadj_results/kappa_154.csv
Columns: circuit_id, N, n_free, kappa

Run on the Linux server:
  python ~/craft_release/craft/cadj/compute_kappa_154.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

_HERE         = os.path.dirname(os.path.abspath(__file__))   # cadj/
_CLAUDE_FILES = os.path.dirname(_HERE)                        # craft/
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)               # craft_release/

for _p in (_CLAUDE_FILES, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parse import parse_netlist  # noqa: E402 — after sys.path setup

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl"),
)
OUT_CSV = os.path.join(_CLAUDE_FILES, "cadj_results", "kappa_154.csv")


def _conductance_matrix(pc) -> np.ndarray:
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g; A[a, b] -= g; A[b, a] -= g
    return A


def compute_kappa(c: dict) -> dict:
    pc  = parse_netlist(c["Netlist"])
    N   = pc.num_nodes
    free  = [i for i in range(N) if not pc.is_fixed[i]]
    fi    = np.array(free)
    A     = _conductance_matrix(pc)
    A_FF  = A[np.ix_(fi, fi)]
    kappa = float(np.linalg.cond(A_FF, p=2))
    return {
        "circuit_id": c["ID"],
        "N": N,
        "n_free": len(free),
        "kappa": f"{kappa:.6g}",
    }


def main():
    circuits = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for i, c in enumerate(circuits):
        try:
            row = compute_kappa(c)
            rows.append(row)
            print(f"[{i+1:>3}/{len(circuits)}] {row['circuit_id']}  n_free={row['n_free']}  kappa={row['kappa']}", flush=True)
        except Exception as e:
            print(f"[{i+1:>3}/{len(circuits)}] {c['ID']}  ERROR: {e}", flush=True)
            rows.append({"circuit_id": c["ID"], "N": "", "n_free": "", "kappa": "ERROR"})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["circuit_id", "N", "n_free", "kappa"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
