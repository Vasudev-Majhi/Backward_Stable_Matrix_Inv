"""E1 — Backward-error vs forward-error for CRAFT LU inversion.

For each matrix A and CRAFT-computed inverse X̂, computes:
  - Normwise backward error (Rigal-Gaches):
      eta_j = ||e_j - A @ x_hat_j||_inf / (||A||_inf * ||x_hat_j||_inf + 1)
    Theory (Higham Thm 8.5): eta = O(n*eps), FLAT in kappa.
  - Forward error: ||x_hat_j - x_ref_j||_inf   [rises as O(kappa*eps)]
  - Right residual: ||A @ X_hat - I||_inf / (||A||_inf * ||X_hat||_inf)

Runs two sweeps:
  1. Synthetic SPD matrices: N in {10,20,50}, kappa in {1..1e4}
  2. All 154 real circuit Laplacians

Output CSVs:
  results/E1_backward_synth.csv
  results/E1_backward_circuits.csv

Run from ~/craft_release/matrix_inversion/inversion2/ with:
  MINV_USE_STANDARD_CACHE=1 ~/miniforge3/bin/python3 E1_backward_error.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
sys.path.insert(0, HERE)

import _bootstrap  # noqa: F401  (inversion2's _bootstrap — sets up transformer_vm)
from build_lu import build_for_matrix
from runner_lu import invert
from lu_factor import doolittle

# parse_netlist lives in ~/craft_release/craft/ — add AFTER the inversion2
# imports so it does not shadow inversion2's own _bootstrap.py.
sys.path.append(os.path.join(HOME, "craft_release", "craft"))

DATASET = os.path.expanduser(
    "~/craft_release/dataset/circuit_dataset_rv.jsonl"
)
MODEL_DIR = "/tmp/E1_models"
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SYNTH_CSV  = os.path.join(RESULTS_DIR, "E1_backward_synth.csv")
CIRCUIT_CSV = os.path.join(RESULTS_DIR, "E1_backward_circuits.csv")

SYNTH_FIELDS = [
    "n", "kappa_target", "kappa_actual",
    "max_eta", "median_eta",          # normwise backward error (flat in kappa)
    "max_omega",                       # componentwise backward error
    "right_residual",                  # ||A X_hat - I|| / (||A|| ||X_hat||)
    "max_forward_err",                 # forward error (rises with kappa)
    "median_forward_err",
    "build_s", "infer_s", "status",
]
CIRCUIT_FIELDS = [
    "circuit_id", "N", "n_free", "kappa_actual",
    "max_eta", "median_eta",
    "max_omega",
    "right_residual",
    "max_forward_err",
    "median_forward_err",
    "build_s", "infer_s", "status",
]

N_VALUES    = [10, 20, 50]
KAPPA_VALUES = [1, 5, 10, 50, 100, 500, 1000, 5000, 10_000]


# ── helpers ──────────────────────────────────────────────────────────────────

def make_spd_kappa(n: int, kappa: float, seed: int = 0) -> np.ndarray:
    """SPD matrix with exact condition number kappa (same as exp_a_kappa_sweep)."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    if n == 1:
        eigvals = np.array([1.0])
    elif kappa == 1:
        eigvals = np.ones(n)
    else:
        r = kappa ** (1.0 / (n - 1))
        eigvals = r ** np.arange(n, dtype=np.float64)
    A = (Q * eigvals) @ Q.T
    return 0.5 * (A + A.T)


def backward_error_metrics(A: np.ndarray, X_hat: np.ndarray) -> dict:
    """Compute backward + forward error metrics for A and computed inverse X_hat.

    All arithmetic in float64.  Normalisation ensures eta is O(1) in kappa.
    """
    n = A.shape[0]
    A_norm_inf = float(np.linalg.norm(A, ord=np.inf))
    X_norm_inf = float(np.linalg.norm(X_hat, ord=np.inf))

    etas  = []   # normwise backward error per column
    omegas = []  # componentwise backward error per column (Oettli-Prager)

    for j in range(n):
        e_j    = np.zeros(n, dtype=np.float64); e_j[j] = 1.0
        x_hat  = X_hat[:, j]
        r_j    = e_j - A @ x_hat           # residual in float64

        denom_normwise = A_norm_inf * float(np.linalg.norm(x_hat, ord=np.inf)) + 1.0
        eta_j = float(np.linalg.norm(r_j, ord=np.inf)) / denom_normwise
        etas.append(eta_j)

        # Componentwise (Oettli-Prager): omega_j = max_k |r_j_k| / (|A|*|x_hat|+|e_j|)_k
        denom_cw = np.abs(A) @ np.abs(x_hat) + np.abs(e_j)
        denom_cw = np.where(denom_cw == 0, 1e-300, denom_cw)
        omega_j  = float(np.max(np.abs(r_j) / denom_cw))
        omegas.append(omega_j)

    # Right residual: ||A X_hat - I||_inf / (||A||_inf * ||X_hat||_inf)
    AX = A @ X_hat
    right_res_num = float(np.linalg.norm(AX - np.eye(n), ord=np.inf))
    right_residual = right_res_num / (A_norm_inf * X_norm_inf + 1e-300)

    X_ref = np.linalg.inv(A)
    fwd_errs = np.abs(X_hat - X_ref)

    return {
        "max_eta":          float(max(etas)),
        "median_eta":       float(np.median(etas)),
        "max_omega":        float(max(omegas)),
        "right_residual":   right_residual,
        "max_forward_err":  float(fwd_errs.max()),
        "median_forward_err": float(np.median(fwd_errs)),
    }


# ── Sweep 1: synthetic SPD ───────────────────────────────────────────────────

def run_synth_sweep():
    print(f"{'n':>4} {'kappa':>7} {'max_eta':>12} {'max_fwd':>12} "
          f"{'ratio':>10}   status", flush=True)

    with open(SYNTH_CSV, "w", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=SYNTH_FIELDS)
        w.writeheader()

        for n in N_VALUES:
            for kappa in KAPPA_VALUES:
                seed = n * 10_000 + int(kappa)
                row: dict = {"n": n, "kappa_target": kappa}
                try:
                    A = make_spd_kappa(n, kappa, seed=seed)
                    row["kappa_actual"] = float(np.linalg.cond(A))

                    t0 = time.time()
                    info = build_for_matrix(A, model_dir=MODEL_DIR)
                    row["build_s"] = round(time.time() - t0, 3)

                    t0 = time.time()
                    X_hat = invert(A, info["model_path"])
                    row["infer_s"] = round(time.time() - t0, 3)

                    metrics = backward_error_metrics(A, X_hat)
                    row.update(metrics)
                    row["status"] = "ok"

                    ratio_str = "N/A" if metrics["max_eta"] == 0 else \
                        f"{metrics['max_forward_err']/metrics['max_eta']:.1f}x"
                    print(f"{n:>4} {kappa:>7} {metrics['max_eta']:>12.3e} "
                          f"{metrics['max_forward_err']:>12.3e} {ratio_str:>10}   ok",
                          flush=True)

                except Exception as exc:
                    row["status"] = f"error:{exc}"
                    row.update({k: "" for k in SYNTH_FIELDS
                                if k not in row})
                    print(f"{n:>4} {kappa:>7}  ERROR: {exc}", flush=True)

                w.writerow({k: row.get(k, "") for k in SYNTH_FIELDS})

    print(f"\nSynth sweep done → {SYNTH_CSV}", flush=True)


# ── Sweep 2: 154 real circuits ───────────────────────────────────────────────

def load_circuits():
    circuits = []
    with open(DATASET) as f:
        for line in f:
            circuits.append(json.loads(line))
    return circuits


def build_aff(netlist_str: str):
    # Inline parse (avoid cadj dependency)
    try:
        from parse import parse_netlist  # type: ignore
    except ImportError:
        raise

    pc = parse_netlist(netlist_str)
    N  = pc.num_nodes
    A  = np.zeros((N, N), dtype=np.float64)
    for (a, b, ohms) in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g
        A[a, b] -= g; A[b, a] -= g
    free = [i for i in range(N) if not pc.is_fixed[i]]
    A_FF = A[np.ix_(free, free)]
    return A_FF, len(free), N


def run_circuit_sweep():
    circuits = load_circuits()
    print(f"\nCircuit sweep: {len(circuits)} circuits", flush=True)
    print(f"{'id':>12} {'n_free':>6} {'kappa':>9} {'max_eta':>12} "
          f"{'max_fwd':>12}   status", flush=True)

    with open(CIRCUIT_CSV, "w", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=CIRCUIT_FIELDS)
        w.writeheader()

        for c in circuits:
            cid = c["ID"]
            row: dict = {"circuit_id": cid}
            try:
                A_FF, n_free, N_total = build_aff(c["Netlist"])
                row["N"] = N_total
                row["n_free"] = n_free
                row["kappa_actual"] = float(np.linalg.cond(A_FF))

                t0 = time.time()
                info = build_for_matrix(A_FF, model_dir=MODEL_DIR)
                row["build_s"] = round(time.time() - t0, 3)

                t0 = time.time()
                X_hat = invert(A_FF, info["model_path"])
                row["infer_s"] = round(time.time() - t0, 3)

                metrics = backward_error_metrics(A_FF, X_hat)
                row.update(metrics)
                row["status"] = "ok"

                print(f"{cid:>12} {n_free:>6} {row['kappa_actual']:>9.1e} "
                      f"{metrics['max_eta']:>12.3e} "
                      f"{metrics['max_forward_err']:>12.3e}   ok", flush=True)

            except Exception as exc:
                row["status"] = f"error:{exc}"
                row.update({k: "" for k in CIRCUIT_FIELDS if k not in row})
                print(f"{cid:>12}  ERROR: {exc}", flush=True)

            w.writerow({k: row.get(k, "") for k in CIRCUIT_FIELDS})

    print(f"\nCircuit sweep done → {CIRCUIT_CSV}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--synth-only",   action="store_true")
    p.add_argument("--circuit-only", action="store_true")
    args = p.parse_args()

    if not args.circuit_only:
        run_synth_sweep()
    if not args.synth_only:
        run_circuit_sweep()
