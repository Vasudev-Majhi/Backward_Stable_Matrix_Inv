"""Build per-matrix iJacobi-style inversion transformer weights.

For each input matrix A, produces:
  model_<hash>_minv.bin          — transformer weights
  model_<hash>_minv.bin.slots.json — sidecar with slot indices and metadata

Usage:
    python build.py --matrix A.npy
    python build.py --matrix A.npy --T 50
    python build.py --matrix A.npy --plan-only
    python build.py --matrix A.npy --out custom.bin
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def matrix_hash(A: np.ndarray) -> str:
    """Stable 12-char hex hash of matrix bytes."""
    return hashlib.sha1(A.tobytes()).hexdigest()[:12]


def build_for_matrix(
    A: np.ndarray,
    T: int | None = None,
    out_path: str | None = None,
    plan_only: bool = False,
    model_dir: str | None = None,
):
    """Build transformer weights for matrix A.

    Returns a dict with model_path, slot indices, and model stats.
    """
    from transformer_vm.model.weights import build_model, save_weights
    from transformer_vm.scheduler.milp import milp_schedule

    from minv_interpreter import MinvMachine
    from minv_reference import auto_T

    n = A.shape[0]
    if T is None:
        T = auto_T(n, A)

    h = matrix_hash(A)
    log.info("Matrix hash=%s  n=%d  T=%d", h, n, T)

    machine = MinvMachine(A, T)

    t0 = time.time()
    pg, meta = machine.build()
    log.info("Graph built in %.2fs: %d dims, %d lookups",
              time.time() - t0, len(pg.all_dims), len(pg.all_lookups))

    if plan_only:
        t1 = time.time()
        sched = milp_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
        log.info("MILP done in %.2fs: n_layers=%d, d_model=%d",
                  time.time() - t1, sched["num_layers"], sched["width"])
        return sched

    t1 = time.time()
    model, all_tokens, tok_to_idx_map, shared = build_model(program_graph=pg)
    log.info("Weights built in %.2fs", time.time() - t1)

    # shared[6] = slot_of (Dimension -> residual slot index)
    slot_of = shared[6]

    # Extract slot indices by dim object identity (not name) — safe under auto_name.
    x_value_slot_dim = meta["x_value_slot_dim"]
    b_value_slot_dim = meta["b_value_slot_dim"]
    x_new_expr = meta["x_new_expr"]
    x_new_dim = next(iter(x_new_expr.terms))

    if x_value_slot_dim not in slot_of:
        raise RuntimeError("x_value_slot dim has no residual slot — MILP layout changed?")
    if b_value_slot_dim not in slot_of:
        raise RuntimeError("b_value_slot dim has no residual slot — MILP layout changed?")
    if x_new_dim not in slot_of:
        # The DSL's persist() returns a user-facing handle dim that is a different
        # Python object from the internal scheduled dim stored in slot_of.
        # Since we call persist() exactly once, there is exactly one PersistDimension
        # in slot_of — that must be x_new. Fall back to type-based search.
        from transformer_vm.graph.core import PersistDimension
        pd_candidates = [(d, s) for d, s in slot_of.items() if isinstance(d, PersistDimension)]
        if not pd_candidates:
            # Last resort: search by name (works if auto_name preserved it)
            pd_candidates = [(d, s) for d, s in slot_of.items()
                             if getattr(d, "name", None) == "x_new"]
        if not pd_candidates:
            raise RuntimeError(
                "x_new persist dim has no slot. slot_of keys: "
                + str([(getattr(d, 'name', None), type(d).__name__) for d in slot_of])
            )
        x_new_dim, _ = pd_candidates[0]
        log.info("x_new slot found via type fallback: name=%r slot=%d",
                 getattr(x_new_dim, "name", None), slot_of[x_new_dim])

    slot_x_value = int(slot_of[x_value_slot_dim])
    slot_b_value = int(slot_of[b_value_slot_dim])
    slot_x_new   = int(slot_of[x_new_dim])
    log.info("Slots: x_value_slot=%d  b_value_slot=%d  x_new=%d",
              slot_x_value, slot_b_value, slot_x_new)

    slot_map_named: dict[str, int] = {}
    for d, s in slot_of.items():
        nm = getattr(d, "name", None)
        if nm:
            slot_map_named[nm] = int(s)

    actual_d_model  = model.tok.weight.shape[1]
    actual_n_layers = len(model.attn)
    actual_d_ffn    = model.ff_in[0].weight.shape[0] // 2
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: d_model=%d, n_layers=%d, d_ffn=%d, vocab=%d, params=%s",
              actual_d_model, actual_n_layers, actual_d_ffn, len(all_tokens),
              f"{n_params:,}")

    if out_path is None:
        base = model_dir if model_dir else os.path.dirname(__file__)
        out_path = os.path.join(base, f"model_{h}_minv.bin")

    save_weights(model, all_tokens, out_path)
    log.info("Saved: %s  (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    sidecar_path = out_path + ".slots.json"
    with open(sidecar_path, "w") as f:
        json.dump({
            "slot_x_value_slot": slot_x_value,
            "slot_b_value_slot": slot_b_value,
            "slot_x_new":        slot_x_new,
            "slot_of_named":     slot_map_named,
            "n": n,
            "T": T,
            "matrix_hash": h,
        }, f, indent=2)
    log.info("Saved sidecar: %s", sidecar_path)

    return {
        "matrix_hash": h,
        "n": n,
        "T": T,
        "n_layers": actual_n_layers,
        "d_model": actual_d_model,
        "d_ffn": actual_d_ffn,
        "vocab": len(all_tokens),
        "n_params": n_params,
        "model_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build per-matrix Jacobi transformer")
    parser.add_argument("--matrix", required=True, help="Path to .npy matrix file")
    parser.add_argument("--T", type=int, default=None, help="Iteration count")
    parser.add_argument("--out", type=str, default=None, help="Output .bin path")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory for output files (default: same as this script)")
    parser.add_argument("--plan-only", action="store_true",
                        help="Only run MILP, no weight build")
    args = parser.parse_args()

    A = np.load(args.matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {A.shape}")

    result = build_for_matrix(
        A, T=args.T, out_path=args.out,
        plan_only=args.plan_only, model_dir=args.model_dir,
    )
    if isinstance(result, dict) and "model_path" in result:
        print(f"\nBuild complete: {result}")


if __name__ == "__main__":
    main()
