"""E-NEW-3 — Theorem 1 empirical verification.

For all 154 circuits in circuit_dataset_rv.jsonl:
  - Parse netlist -> conductance matrix A_FF (free-node block)
  - Compute Jacobi iteration matrix M = D^{-1} (D - A_FF) = I - D^{-1} A_FF
  - Compute spectral quantities: rho(M), ||M||_inf, det(I - M)
  - Compute Theorem 1 quantities:
      r_stall = h / (2 * (1 - ||M||_inf))   with h = V_STEP = 0.05
      mink_thr = (1/2)^N
  - Cross-reference with failure_modes_E14.csv (Mode A / B_simple / B_compound / C / PASS)

Output: theorem1_verification.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/craft_release")
sys.path.insert(0, os.path.join(ROOT, "craft"))

from parse import parse_netlist  # noqa: E402

DATASET = os.path.join(ROOT, "dataset", "circuit_dataset_rv.jsonl")
FAILURE_CSV = os.path.join(
    ROOT, "craft/experiments/results/failure_modes_E14.csv"
)
OUT_CSV = os.path.join(
    ROOT, "craft/paper_experiments/results/theorem1_verification.csv"
)

V_STEP = 0.05
TOL = 0.05


def build_AFF(pc):
    """Conductance matrix block A_FF for free nodes."""
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g
        A[b, b] += g
        A[a, b] -= g
        A[b, a] -= g
    free = [i for i, fx in enumerate(pc.is_fixed) if not fx]
    A_FF = A[np.ix_(free, free)]
    return A_FF, free


def compute_M_quantities(A_FF):
    n = A_FF.shape[0]
    if n == 0:
        return dict(n_free=0, rho_M=0.0, norm_M_inf=0.0, det_IminusM=1.0,
                    log_det_abs=0.0, mink_log=0.0, r_stall=0.0)
    D = np.diag(np.diag(A_FF))
    LU = D - A_FF  # off-diagonal contribution; M = D^{-1} (D - A) = I - D^{-1} A
    # iteration matrix
    Dinv = np.diag(1.0 / np.diag(A_FF))
    M = Dinv @ LU
    eig = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(eig)))
    norm_inf = float(np.max(np.sum(np.abs(M), axis=1)))
    IminusM = np.eye(n) - M
    det = float(np.linalg.det(IminusM))
    log_det_abs = float(np.log(abs(det))) if det != 0 else float("-inf")
    mink_log = float(n * np.log(0.5))
    if norm_inf < 1.0:
        r_stall = V_STEP / (2.0 * (1.0 - norm_inf))
    else:
        r_stall = float("inf")
    return dict(
        n_free=n, rho_M=rho, norm_M_inf=norm_inf, det_IminusM=det,
        log_det_abs=log_det_abs, mink_log=mink_log, r_stall=r_stall,
    )


def load_failure_modes():
    """Return dict circuit_id -> failure_mode (PASS / A / B_simple / B_compound / C)."""
    if not os.path.exists(FAILURE_CSV):
        return {}
    out = {}
    with open(FAILURE_CSV) as f:
        for row in csv.DictReader(f):
            mode = row.get("failure_mode") or row.get("mode") or row.get("category", "")
            out[row["circuit_id"]] = mode
    return out


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    failure_modes = load_failure_modes()
    print(f"Loaded failure_mode mapping for {len(failure_modes)} circuits.")

    rows = []
    with open(DATASET) as f:
        for line in f:
            ckt = json.loads(line)
            cid = ckt["ID"]
            try:
                pc = parse_netlist(ckt["Netlist"])
                A_FF, _ = build_AFF(pc)
                quants = compute_M_quantities(A_FF)
                quants["circuit_id"] = cid
                quants["complexity"] = ckt.get("Complexity", "")
                quants["N"] = pc.num_nodes
                quants["failure_mode"] = failure_modes.get(cid, "UNKNOWN")
                # Theorem 1 (c) Minkowski check: |det(I-M)| < (1/2)^N
                quants["minkowski_holds"] = bool(
                    quants["det_IminusM"] != 0
                    and np.log(abs(quants["det_IminusM"])) < quants["mink_log"]
                )
                # Theorem 1 (d) stall radius check: r_stall > tol
                quants["r_stall_exceeds_tol"] = bool(quants["r_stall"] > TOL)
                rows.append(quants)
            except Exception as e:
                print(f"  ERROR {cid}: {e}")
                rows.append({"circuit_id": cid, "error": str(e)})

    fieldnames = [
        "circuit_id", "complexity", "N", "n_free", "failure_mode",
        "rho_M", "norm_M_inf", "det_IminusM", "log_det_abs", "mink_log",
        "minkowski_holds", "r_stall", "r_stall_exceeds_tol", "error",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Verification print: how many B_compound circuits satisfy both predictions?
    bc = [r for r in rows if r.get("failure_mode") == "B_compound"]
    if bc:
        n = len(bc)
        n_mink = sum(1 for r in bc if r.get("minkowski_holds"))
        n_stall = sum(1 for r in bc if r.get("r_stall_exceeds_tol"))
        n_both = sum(
            1 for r in bc
            if r.get("minkowski_holds") and r.get("r_stall_exceeds_tol")
        )
        print()
        print(f"=== Theorem 1 verification on B_compound failures (n={n}) ===")
        print(f"  Minkowski cond  |det(I-M)| < (1/2)^N : {n_mink}/{n}")
        print(f"  Stall radius   r_stall > tol         : {n_stall}/{n}")
        print(f"  Both           (Theorem 1 c & d)     : {n_both}/{n}")
    a = [r for r in rows if r.get("failure_mode") == "A_non_convergent"]
    if a:
        n = len(a)
        rhos = [r.get("rho_M") for r in a if isinstance(r.get("rho_M"), float)]
        if rhos:
            print(f"=== A_non_convergent (n={n}): rho(M) >= 0.997 in {sum(1 for r in rhos if r >= 0.997)}/{n}")
            print(f"     min rho={min(rhos):.6f}  max rho={max(rhos):.6f}")
    print(f"\nWrote {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
