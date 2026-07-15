"""Run a per-matrix LU-decomposition inversion transformer.

Loads model_<hash>_lu.bin (built by build_lu.py) plus its .slots.json sidecar,
injects b column-by-column, performs forward and back substitution as 3n+2
token forward passes, and stitches the result into A^{-1}.

All matrix arithmetic happens inside the transformer's forward passes:
the runner only writes runtime values into specific residual slots and reads
the persisted x_new slot back. No numpy linalg is called at inference time.

Runner contract:
  - On fwd_i:
      * Inject b[i] into slot_b_value.
      * Forward pass produces y_i in slot_x_new (because L[i,i]=1, the
        diagonal scaling is a no-op, and the FFN computes
        b[i] - sum_{j<i} L[i,j] * y_j directly).
      * Save y_buf[i] = x_new.
      * V-cache patch the position so future bck_i tokens fetching this
        position via x_value_slot see y_i, not 0.
  - On bck_i (sequenced in REVERSE order, i = n-1 .. 0):
      * Inject y_buf[i] into slot_y_input.
      * Forward pass produces x_i = (y_i - sum_{j>i} U[i,j]*x_j) / U[i,i]
        in slot_x_new.
      * Save x_result[i].
      * V-cache patch so future bck_{i'} tokens (with i' < i) fetching this
        position via x_value_slot see x_i, not 0.

Usage:
    python runner_lu.py --matrix A.npy [--out X.npy] [--verbose]
    python runner_lu.py --matrix A.npy --model-dir /path/to/models
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

_FORCE_STANDARD = os.environ.get("MINV_USE_STANDARD_CACHE", "0") == "1"
if _FORCE_STANDARD:
    from transformer_vm.attention.standard_cache import StandardKVCache as _DefaultCache
    USING_HULL = False
else:
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

    Snapshots x[slot] AFTER each FFN layer if snapshot_slots is given.
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


def _vcache_patch_lu(
    model, cache, x_init: torch.Tensor, pos: int, n_layers: int,
    slot_x_value: int, slot_x_new: int, new_value: float,
    extra: dict[int, float],
):
    """Overwrite this position's V cache so future tokens see the freshly-computed
    value (y_i during fwd, x_i during bck) rather than the pre-FFN zero in
    slot_x_value.

    extra: {slot_idx: value} for any other slots that should also be set
    (slot_b_value during fwd, slot_y_input during bck) -- mirrors what was in
    x_init but ensures correctness even if MILP co-locates slots.

    CRITICAL: slot_x_new MUST be set LAST so it overrides any colliding slot.
    See runner.py:144 comment about the n=6 slot-collision V-cache bug.
    """
    x_corrected = x_init.clone()
    x_corrected[slot_x_value] = float(new_value)
    for slot_idx, value in extra.items():
        x_corrected[slot_idx] = float(value)
    x_corrected[slot_x_new] = float(new_value)   # MUST be last (slot collision safety)

    for li in range(n_layers):
        new_kqv = model.attn[li].in_proj_weight @ x_corrected
        new_k, _, new_v = new_kqv.chunk(3, dim=-1)
        # Two cache classes are supported here:
        #   StandardKVCache: literal in-place V replacement on _vals[layer][-1].
        #   HullKVCache:     insert_v rewrites the entry's metadata at the
        #                    most recent seq for this layer (no append).
        if hasattr(cache, "_vals"):
            cache._vals[li][-1] = new_v.clone()
        elif hasattr(cache, "insert_v"):
            cache.insert_v(li, new_k, new_v)


def solve_column_lu(
    model,
    tok_to_idx: dict[str, int],
    n: int,
    slot_x_value: int,
    slot_b_value: int,
    slot_y_input: int,
    slot_x_new: int,
    b: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Run one inference for RHS b. Returns x ~= A^{-1} * b.

    The transformer performs forward+back substitution; this function only
    threads runtime values between tokens.
    """
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = CACHE_CLASS(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    y_buf = [0.0] * n
    x_result = [0.0] * n
    pos = 0

    tokens = (
        ["start"]
        + [f"init_{i}" for i in range(n)]
        + [f"fwd_{i}" for i in range(n)]
        + [f"bck_{i}" for i in range(n - 1, -1, -1)]
        + ["halt"]
    )

    with torch.no_grad():
        for tok in tokens:
            tok_idx = tok_to_idx[tok]
            x_init = _build_x(model, tok_idx, pos)

            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_b_value] = x_init[slot_b_value] + float(b[i])

                x, snaps = _forward_step(
                    model, cache, x_init, pos, snapshot_slots=[slot_x_new]
                )
                snap_vals = [s[0] for s in snaps] if snaps else [0.0]
                y_buf[i] = snap_vals[-1]

                if verbose:
                    log.info("  pos %d %s  y_new=%.10f", pos, tok, y_buf[i])

                _vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value,
                    slot_x_new=slot_x_new,
                    new_value=y_buf[i],
                    extra={slot_b_value: float(b[i])},
                )

            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_y_input] = x_init[slot_y_input] + float(y_buf[i])

                x, snaps = _forward_step(
                    model, cache, x_init, pos, snapshot_slots=[slot_x_new]
                )
                snap_vals = [s[0] for s in snaps] if snaps else [0.0]
                x_result[i] = snap_vals[-1]

                if verbose:
                    log.info("  pos %d %s  x_new=%.10f", pos, tok, x_result[i])

                _vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value,
                    slot_x_new=slot_x_new,
                    new_value=x_result[i],
                    extra={slot_y_input: float(y_buf[i])},
                )
            else:
                # init_*, start, halt -- no injection, no V-cache patch.
                x, _ = _forward_step(model, cache, x_init, pos)

            pos += 1

    return np.array(x_result)


def invert(
    A: np.ndarray,
    model_path: str,
    verbose: bool = False,
) -> np.ndarray:
    """Invert A using the pre-built LU transformer at model_path.

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
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])
    n_side = int(sidecar["n"])

    n = A.shape[0]
    if n != n_side:
        raise ValueError(f"Matrix n={n} does not match sidecar n={n_side}")

    model, all_tokens, tok_to_idx = load_weights(model_path)
    model.eval()

    log.info("Inverting %dx%d matrix (LU), slots x_val=%d b_val=%d y_in=%d x_new=%d",
              n, n, slot_x_value, slot_b_value, slot_y_input, slot_x_new)

    X = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0
        if verbose:
            log.info("Column j=%d", j)
        X[:, j] = solve_column_lu(
            model, tok_to_idx, n,
            slot_x_value, slot_b_value, slot_y_input, slot_x_new,
            b=e_j, verbose=verbose,
        )

    return X


def main():
    parser = argparse.ArgumentParser(description="Run per-matrix LU inversion transformer")
    parser.add_argument("--matrix", required=True, help="Path to .npy matrix file")
    parser.add_argument("--out", type=str, default=None, help="Output .npy path for X = A^{-1}")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing model_<hash>_lu.bin")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import hashlib
    A = np.load(args.matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {A.shape}")

    h = hashlib.sha1(A.tobytes()).hexdigest()[:12]
    model_dir = args.model_dir if args.model_dir else os.path.dirname(__file__)
    model_path = os.path.join(model_dir, f"model_{h}_lu.bin")

    t0 = time.time()
    X = invert(A, model_path, verbose=args.verbose)
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
