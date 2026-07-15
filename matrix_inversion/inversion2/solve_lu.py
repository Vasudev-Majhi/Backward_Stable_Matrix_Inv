"""Solve A·x = b using the pre-built LU transformer.

Loads model_<hash>_lu.bin + .slots.json sidecar (produced by build_lu.py),
runs a single forward+back substitution pass, and saves x to --out.

All arithmetic happens inside the transformer's forward passes.
No numpy.linalg is called.

Usage:
    python solve_lu.py --matrix A.npy --b b.npy --model-dir /path --out x.npy
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import logging
import os
import time

import numpy as np

from runner_lu import solve_column_lu, CACHE_CLASS  # noqa: F401 (CACHE_CLASS selected at import)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def solve(
    A: np.ndarray,
    b: np.ndarray,
    model_dir: str,
    verbose: bool = False,
) -> np.ndarray:
    """Return x ≈ A⁻¹·b via the LU transformer. No linalg called."""
    from transformer_vm.model.weights import load_weights

    h = hashlib.sha1(A.tobytes()).hexdigest()[:12]
    model_path = os.path.join(model_dir, f"model_{h}_lu.bin")
    sidecar_path = model_path + ".slots.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LU model not found: {model_path}")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])
    n_side       = int(sidecar["n"])

    n = A.shape[0]
    if n != n_side:
        raise ValueError(f"Matrix n={n} does not match sidecar n={n_side}")
    if b.shape != (n,):
        raise ValueError(f"b shape {b.shape} must be ({n},)")

    model, _, tok_to_idx = load_weights(model_path)
    model.eval()

    log.info("Solving %dx%d system via LU transformer", n, n)
    t0 = time.time()
    x = solve_column_lu(
        model, tok_to_idx, n,
        slot_x_value, slot_b_value, slot_y_input, slot_x_new,
        b=b, verbose=verbose,
    )
    log.info("Solve done in %.3fs", time.time() - t0)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve A·x=b via LU transformer")
    ap.add_argument("--matrix",    required=True, help=".npy file for A")
    ap.add_argument("--b",         required=True, help=".npy file for b vector")
    ap.add_argument("--model-dir", required=True, help="Directory with model_<hash>_lu.bin")
    ap.add_argument("--out",       required=True, help="Output .npy path for x")
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    A = np.load(args.matrix)
    b = np.load(args.b)
    x = solve(A, b, args.model_dir, verbose=args.verbose)
    np.save(args.out, x)


if __name__ == "__main__":
    main()
