"""Run a per-matrix Jacobi inversion transformer.

Loads model_<hash>_minv.bin (built by build.py) plus its .slots.json sidecar,
injects b and x column-by-column, and stitches the result into A^{-1}.

Runner contract:
  - Per-column buffer x_buf[i], initialized to 0.
  - Before up_i forward pass: x[slot_x_value] += x_buf[i], x[slot_b_value] += b[i].
  - After  up_i forward pass: x_buf[i] = max-magnitude snapshot of x_new across layers.
  - V-cache patch after every up_i (§2.3 of plan): overwrite the last V cache entry
    so future tokens see the freshly-computed x_new, not the stale pre-forward value.
    Without this the recurrence has a 2-step lag and Jacobi converges at half rate.

Usage:
    python runner.py --matrix A.npy [--out X.npy] [--T 30] [--verbose]
    python runner.py --matrix A.npy --model-dir /path/to/models
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

import os as _os
_local_bin = _os.path.expanduser("~/.local/bin")
if _local_bin not in _os.environ.get("PATH", ""):
    _os.environ["PATH"] = _local_bin + _os.pathsep + _os.environ.get("PATH", "")

try:
    from transformer_vm.attention.hull_cache import HullKVCache as _DefaultCache
    USING_HULL = True
except Exception:
    from transformer_vm.attention.standard_cache import StandardKVCache as _DefaultCache
    USING_HULL = False
CACHE_CLASS = _DefaultCache


def _forward_step(
    model, cache, x_initial: torch.Tensor, pos: int,
    snapshot_slots: list[int] | None = None,
) -> tuple[torch.Tensor, list[list[float]] | None]:
    """Run one token's forward pass. Returns (final_x, snapshots).

    If snapshot_slots is given, snapshots x[slot] AFTER each FFN layer
    (n_layers entries total). Capturing after FFN avoids reading the static
    embedding value (e.g. is_update_tok=1) that MILP may co-locate at the
    same slot as x_new.
    """
    import torch.nn.functional as F

    x = x_initial.clone()
    snapshots: list[list[float]] | None = [] if snapshot_slots is not None else None

    for layer_idx, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
        out = cache.layer_step(layer_idx, k, q, v)
        x = x + attn.out_proj(out)

        gate, val = ff_in(x).chunk(2, dim=-1)
        x = x + ff_out(F.relu(gate) * val)

        if snapshots is not None:
            snapshots.append([float(x[s].item()) for s in snapshot_slots])

    return x, snapshots


def _build_x(model, tok_idx: int, pos: int) -> torch.Tensor:
    """Token embedding + positional encoding, no runtime injection."""
    from transformer_vm.model.transformer import add_position_encoding

    x = model.tok.weight[tok_idx].clone()
    add_position_encoding(x, pos)
    return x


def solve_column(
    model,
    tok_to_idx: dict[str, int],
    n: int,
    T: int,
    slot_x_value: int,
    slot_b_value: int,
    slot_x_new: int,
    b: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Run one inference for RHS b. Returns x ≈ A^{-1} · b."""
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = CACHE_CLASS(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    x_buf = [0.0] * n
    pos = 0

    tokens = (
        ["start"]
        + [f"init_{i}" for i in range(n)]
        + [f"up_{i}" for _ in range(T) for i in range(n)]
        + ["halt"]
    )

    with torch.no_grad():
        for tok in tokens:
            tok_idx = tok_to_idx[tok]
            x_init = _build_x(model, tok_idx, pos)

            if tok.startswith("up_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_x_value] = x_init[slot_x_value] + float(x_buf[i])
                x_init[slot_b_value] = x_init[slot_b_value] + float(b[i])

                x, snaps = _forward_step(
                    model, cache, x_init, pos, snapshot_slots=[slot_x_new]
                )
                snap_vals = [s[0] for s in snaps] if snaps else [0.0]
                x_buf[i] = snap_vals[-1]  # persist contract: x_new is stable in final FFN output

                if verbose:
                    log.info("  pos %d %s  x_new=%.8f", pos, tok, x_buf[i])

                # V-cache patch: overwrite this position's V so future tokens
                # see the freshly-computed x_new, not the stale pre-forward value.
                x_corrected = x_init.clone()
                x_corrected[slot_x_value] = float(x_buf[i])
                x_corrected[slot_b_value] = float(b[i])
                x_corrected[slot_x_new] = float(x_buf[i])  # overrides b[i] when slot_x_new==slot_b_value
                for li in range(n_layers):
                    new_kqv = model.attn[li].in_proj_weight @ x_corrected
                    new_k, _, new_v = new_kqv.chunk(3, dim=-1)
                    if hasattr(cache, "_vals"):
                        cache._vals[li][-1] = new_v.clone()
                    elif hasattr(cache, "update_v_at"):
                        cache.update_v_at(li, pos - 1, new_v)
                    elif hasattr(cache, "insert_v"):
                        cache.insert_v(li, new_k, new_v)
            else:
                x, _ = _forward_step(model, cache, x_init, pos)

            pos += 1

    return np.array(x_buf)


def invert(
    A: np.ndarray,
    model_path: str,
    T: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Invert A using the pre-built transformer at model_path.

    Loads the .slots.json sidecar automatically.
    """
    from transformer_vm.model.weights import load_weights

    sidecar_path = model_path + ".slots.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])
    n_side = int(sidecar["n"])
    T_side = int(sidecar["T"])

    if T is None:
        T = T_side

    n = A.shape[0]
    if n != n_side:
        raise ValueError(f"Matrix n={n} does not match sidecar n={n_side}")

    model, all_tokens, tok_to_idx = load_weights(model_path)
    model.eval()

    log.info("Inverting %dx%d matrix, T=%d, slots x_val=%d b_val=%d x_new=%d",
              n, n, T, slot_x_value, slot_b_value, slot_x_new)

    X = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0
        if verbose:
            log.info("Column j=%d", j)
        X[:, j] = solve_column(
            model, tok_to_idx, n, T,
            slot_x_value, slot_b_value, slot_x_new,
            b=e_j, verbose=verbose,
        )

    return X


def main():
    parser = argparse.ArgumentParser(description="Run per-matrix Jacobi inversion transformer")
    parser.add_argument("--matrix", required=True, help="Path to .npy matrix file")
    parser.add_argument("--out", type=str, default=None, help="Output .npy path for X = A^{-1}")
    parser.add_argument("--T", type=int, default=None,
                        help="Override iteration count (default: from sidecar)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing model_<hash>_minv.bin")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import hashlib
    A = np.load(args.matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {A.shape}")

    h = hashlib.sha1(A.tobytes()).hexdigest()[:12]
    model_dir = args.model_dir if args.model_dir else os.path.dirname(__file__)
    model_path = os.path.join(model_dir, f"model_{h}_minv.bin")

    t0 = time.time()
    X = invert(A, model_path, T=args.T, verbose=args.verbose)
    elapsed = time.time() - t0

    Xref = np.linalg.inv(A)
    max_err = float(np.max(np.abs(X - Xref)))
    med_err = float(np.median(np.abs(X - Xref)))
    log.info("Done in %.2fs | max_abs_err=%.2e | median_abs_err=%.2e",
              elapsed, max_err, med_err)

    if args.out:
        np.save(args.out, X)
        log.info("Saved X to %s", args.out)
    else:
        print("Predicted X:")
        print(X)
        print("numpy ref:")
        print(Xref)
        print(f"max_abs_err={max_err:.2e}  median_abs_err={med_err:.2e}")


if __name__ == "__main__":
    main()
