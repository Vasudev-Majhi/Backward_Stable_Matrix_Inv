"""Build per-circuit sidecar for the idea3 end-to-end LU inference pipeline.

Pipeline (build time, once per circuit):
  1. Extract A_FF, A_FP from the conductance matrix.
  2. Build the LU transformer for A_FF via subprocess -> inversion2/build_lu.py.
  3. Save a sidecar {A_FF.npy, A_FP.npy, free_indices, fixed_indices,
     target_free_idx, lu_model_path} to idea3/sidecars/<CID>/.

Paper claim: no numpy.linalg.solve or numpy.linalg.inv is called here.
A_FF^-1 is NEVER computed at build time — the LU transformer runs at inference time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

_REPO_ROOT = os.path.dirname(_CLAUDE_FILES)

_INVERSION2_CANDIDATES = [
    os.path.join(os.path.dirname(_REPO_ROOT), "Matrix_inversion", "inversion2"),  # Windows
    os.path.join(_REPO_ROOT, "matrix_inversion", "inversion2"),                    # server
]
INVERSION2_DIR = next(
    (p for p in _INVERSION2_CANDIDATES if os.path.isdir(p)),
    _INVERSION2_CANDIDATES[0],
)
PYTHON_EXE = sys.executable

_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl")
DATASET_PATH = os.environ.get("CRAFT_DATASET", _DEFAULT_DATASET)

SIDECAR_DIR = os.path.join(_HERE, "sidecars")


def load_circuit(cid: str) -> dict:
    import json as _json
    with open(DATASET_PATH) as f:
        for line in f:
            c = _json.loads(line)
            if c["ID"] == cid:
                return c
    raise KeyError(f"Circuit {cid} not found in {DATASET_PATH}")


def _conductance_matrix(pc) -> np.ndarray:
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g; A[a, b] -= g; A[b, a] -= g
    return A


def _lu_model_path(A_FF: np.ndarray, cid_lu_dir: str) -> str:
    h = hashlib.sha1(A_FF.tobytes()).hexdigest()[:12]
    return os.path.join(cid_lu_dir, f"model_{h}_lu.bin")


def build_lu_transformer(A_FF: np.ndarray, cid: str, lu_dir: str) -> str:
    """Build LU transformer for A_FF. Returns model_path. Cached by hash."""
    os.makedirs(lu_dir, exist_ok=True)
    model_path = _lu_model_path(A_FF, lu_dir)

    if os.path.exists(model_path) and os.path.exists(model_path + ".slots.json"):
        log.info("LU model cached: %s", model_path)
        return model_path

    build_script = os.path.join(INVERSION2_DIR, "build_lu.py")
    r = None
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        A_FF_path = tf.name
    try:
        np.save(A_FF_path, A_FF)
        log.info("Building LU transformer for %s (n=%d) ...", cid, A_FF.shape[0])
        t0 = time.time()
        r = subprocess.run(
            [PYTHON_EXE, build_script,
             "--matrix", A_FF_path,
             "--model-dir", lu_dir],
            cwd=INVERSION2_DIR,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"build_lu.py failed for {cid} (exit {r.returncode}):\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        log.info("LU build done in %.1fs", time.time() - t0)
    finally:
        try:
            os.unlink(A_FF_path)
        except OSError:
            pass

    if not os.path.exists(model_path):
        stdout = r.stdout if r is not None else ""
        raise RuntimeError(
            f"Expected model not found after build: {model_path}\n"
            f"build_lu.py stdout:\n{stdout}"
        )
    return model_path


def build_for_circuit(
    cid: str,
    force: bool = False,
) -> dict:
    """Build idea3 sidecar for cid. No A_FF^-1, no S, no readout model."""
    from parse import parse_netlist

    sidecar_dir = os.path.join(SIDECAR_DIR, cid)
    sidecar_json = os.path.join(sidecar_dir, "sidecar.json")

    if os.path.exists(sidecar_json) and not force:
        log.info("Sidecar cached: %s", sidecar_json)
        with open(sidecar_json) as f:
            return json.load(f)

    c = load_circuit(cid)
    netlist = c["Netlist"]
    target  = int(c["Target_Node"])

    pc = parse_netlist(netlist)
    N  = pc.num_nodes

    A     = _conductance_matrix(pc)
    free  = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]

    if not free or not fixed:
        raise ValueError(f"{cid}: degenerate (free={len(free)}, fixed={len(fixed)})")

    os.makedirs(sidecar_dir, exist_ok=True)

    if target in fixed:
        info = {
            "circuit": cid, "N": N,
            "n_free": len(free), "n_fixed": len(fixed),
            "free_indices": free, "fixed_indices": fixed,
            "target_node": target, "target_is_fixed": True,
            "target_fixed_voltage": pc.fixed_voltage[target],
        }
        with open(sidecar_json, "w") as f:
            json.dump(info, f, indent=2)
        log.info("Sidecar saved (fixed target): %s", sidecar_json)
        return info

    target_free_idx = free.index(target)

    fi, pi = np.array(free), np.array(fixed)
    A_FF = A[np.ix_(fi, fi)]
    A_FP = A[np.ix_(fi, pi)]

    lu_dir = os.path.join(sidecar_dir, "lu_model")
    lu_model_path = build_lu_transformer(A_FF, cid, lu_dir)

    A_FF_path = os.path.join(sidecar_dir, "A_FF.npy")
    A_FP_path = os.path.join(sidecar_dir, "A_FP.npy")
    np.save(A_FF_path, A_FF)
    np.save(A_FP_path, A_FP)

    info = {
        "circuit": cid, "N": N,
        "n_free": len(free), "n_fixed": len(fixed),
        "free_indices": free, "fixed_indices": fixed,
        "target_node": target, "target_is_fixed": False,
        "target_free_idx": target_free_idx,
        "lu_model_path": lu_model_path,
        "A_FF_path": A_FF_path, "A_FP_path": A_FP_path,
    }
    with open(sidecar_json, "w") as f:
        json.dump(info, f, indent=2)

    log.info("Sidecar saved: %s", sidecar_json)
    return info


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build idea3 sidecar (LU transformer, no S)")
    ap.add_argument("circuit_id", help="e.g. CKT_0001")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    result = build_for_circuit(args.circuit_id, force=args.force)
    print(f"\nBuild complete: {result}")


if __name__ == "__main__":
    main()
