"""Run circuit inference using the idea3 end-to-end LU pipeline (server version).

Inference: b = -A_FP @ V_fixed (CPU), then A_FF·x=b via LU transformer subprocess.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

_INVERSION2_CANDIDATES = [
    os.path.join(_REPO_ROOT, "matrix_inversion", "inversion2"),
    os.path.join(os.path.dirname(_REPO_ROOT), "Matrix_inversion", "inversion2"),
]
INVERSION2_DIR = next(
    (p for p in _INVERSION2_CANDIDATES if os.path.isdir(p)),
    _INVERSION2_CANDIDATES[0],
)

_VENV_PYTHON = os.path.join(_REPO_ROOT, "venv", "bin", "python3")
PYTHON_EXE   = _VENV_PYTHON if os.path.isfile(_VENV_PYTHON) else sys.executable

_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl")
DATASET_PATH = os.environ.get("CRAFT_DATASET", _DEFAULT_DATASET)

SIDECAR_DIR = os.path.join(_HERE, "sidecars")


def _load_circuit(cid: str) -> dict:
    import json as _json
    with open(DATASET_PATH) as f:
        for line in f:
            c = _json.loads(line)
            if c["ID"] == cid:
                return c
    raise KeyError(f"Circuit {cid} not found")


def run_circuit_idea3(
    cid: str,
    tol: float = 0.05,
    verbose: bool = False,
) -> tuple[str, float, float, float]:
    """Return (status, pred_v, truth_v, elapsed). LU solve at inference time."""
    from parse import parse_netlist

    sidecar_json = os.path.join(SIDECAR_DIR, cid, "sidecar.json")
    if not os.path.exists(sidecar_json):
        return "NO_SIDECAR", 0.0, 0.0, 0.0

    with open(sidecar_json) as f:
        sc = json.load(f)

    c     = _load_circuit(cid)
    truth = float(c["Ground_Truth_Vout"])

    # Fixed-target shortcut: voltage is directly known, no solve needed.
    if sc.get("target_is_fixed"):
        pred_v = float(sc["target_fixed_voltage"])
        status = "PASS" if abs(pred_v - truth) <= tol else "FAIL"
        return status, pred_v, truth, 0.0

    pc = parse_netlist(c["Netlist"])

    fixed_indices   = sc["fixed_indices"]
    target_free_idx = sc["target_free_idx"]
    A_FP_path       = sc["A_FP_path"]
    A_FF_path       = sc["A_FF_path"]
    lu_model_path   = sc["lu_model_path"]
    lu_dir          = os.path.dirname(lu_model_path)

    A_FP    = np.load(A_FP_path)
    A_FF    = np.load(A_FF_path)
    V_fixed = np.array([pc.fixed_voltage[i] for i in fixed_indices], dtype=np.float64)
    b       = -A_FP @ V_fixed   # RHS vector (pure matmul, no linalg)

    t0 = time.time()

    # Step 1: compute A_FF^-1 explicitly inside the LU transformer.
    # runner_lu.py runs n solves (one per column e_j) and returns the full inverse.
    runner_script = os.path.join(INVERSION2_DIR, "runner_lu.py")
    with tempfile.TemporaryDirectory() as tmp:
        A_FF_inv_path = os.path.join(tmp, "A_FF_inv.npy")

        r = subprocess.run(
            [PYTHON_EXE, runner_script,
             "--matrix",   A_FF_path,
             "--model-dir", lu_dir,
             "--out",       A_FF_inv_path],
            cwd=INVERSION2_DIR,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"runner_lu.py failed for {cid} (exit {r.returncode}):\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )

        A_FF_inv = np.load(A_FF_inv_path)

    # Step 2: V_free = A_FF^-1 @ b  (plain matmul, no linalg)
    V_free = A_FF_inv @ b
    pred_v = float(V_free[target_free_idx])
    elapsed = time.time() - t0
    status  = "PASS" if abs(pred_v - truth) <= tol else "FAIL"
    return status, pred_v, truth, elapsed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit_id")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    status, pred_v, truth, elapsed = run_circuit_idea3(
        args.circuit_id, tol=args.tol, verbose=args.verbose,
    )
    err = abs(pred_v - truth)
    print(f"{status}  {args.circuit_id}  pred={pred_v:.6f}  truth={truth:.6f}  "
          f"err={err:.6f}  time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
