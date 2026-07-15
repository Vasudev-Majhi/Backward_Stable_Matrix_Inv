"""
Compression training loop (Tracr §5 protocol).

Loss: L = L_out + lambda_layer * L_layer
  L_out:   MSE between compressed-model final readout and base-model readout
           (or argmax-token cross-entropy on the readout position).
  L_layer: per-layer reconstruction MSE between
           expand(x_c_layer) and base_model_residual_at_layer.

Optimizer: AdamW, weight_decay=0.1, betas=(0.9, 0.99), 3e5 steps, batch 256,
linear LR decay 1e-3 -> 1e-6 (Tracr defaults).

OUTSTANDING DECISIONS (see paper_data.md §6 T3):
  Q1: which compiled model is the subject?
  Q2: which circuit(s)?
  Q3: where are the .bin weights (local vs server)?
  Q4: compute environment?
  Q5: float32 ok for the wrapper?
  Q6: KV cache off during training?

This file is a SCAFFOLD. Replace the load_compiled_model() and load_circuit_inputs()
stubs with the actual loaders for whichever compiled algorithm we choose.
"""

from __future__ import annotations
import dataclasses
import math
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F

from compressed_transformer import CompressedTransformer  # local import


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclasses.dataclass
class TrainCfg:
    circuit_id: str = "CKT_0001"
    algorithm: str = "lu_direct"   # or "jacobi", "ijacob", "direct"
    d_compressed: int = 16          # sweep target dim
    n_steps: int = 300_000
    batch_size: int = 256
    lr_max: float = 1e-3
    lr_min: float = 1e-6
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.99)
    lambda_layer: float = 1.0       # weight on per-layer recon loss
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32   # Tracr §5; differs from base float64
    out_dir: str = "results"
    log_every: int = 1000


# ============================================================================
# STUBS (to be filled once Q1–Q3 are answered)
# ============================================================================
def load_compiled_model(cfg: TrainCfg):
    """Load the per-circuit compiled VanillaTransformer.

    For LU-direct (current default subject), we expect:
      ../CRAFT/lu_pipeline/<???>/model_<CKT>_lu_direct.bin
      ../CRAFT/lu_pipeline/<???>/model_<CKT>_lu_direct.bin.slots.json

    The exact path layout depends on Q3 (local vs server). For Jacobi compiled
    models, see craft/. For iJacobi, see ijacob_hull/ (note the runner-
    side V-cache patch — compression of iJacobi is harder; see paper_data.md).
    """
    raise NotImplementedError(
        "load_compiled_model: blocked on Q1 (subject), Q2 (circuit), Q3 (path). "
        "See paper_data.md §6 T3."
    )


def load_circuit_inputs(cfg: TrainCfg):
    """Load the input token sequence for the circuit + the ground-truth readout.

    Returns: (token_ids: List[int], readout_position: int, ground_truth_token_id: int).
    """
    raise NotImplementedError(
        "load_circuit_inputs: blocked on Q2 (circuit identity)."
    )


# ============================================================================
# LOSS
# ============================================================================
def compute_layer_loss(
    layer_residuals_compressed_full: List[torch.Tensor],
    layer_residuals_base: List[torch.Tensor],
) -> torch.Tensor:
    if len(layer_residuals_compressed_full) != len(layer_residuals_base):
        raise ValueError("layer count mismatch")
    losses = [
        F.mse_loss(c, b) for c, b in
        zip(layer_residuals_compressed_full, layer_residuals_base, strict=True)
    ]
    return torch.stack(losses).mean()


def compute_out_loss(
    compressed_logits: torch.Tensor, base_logits: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(compressed_logits, base_logits)


# ============================================================================
# GROUND-TRUTH RESIDUAL CAPTURE (run base model once, store per-layer residuals)
# ============================================================================
@torch.no_grad()
def capture_base_residuals(
    base_model, token_ids: List[int], readout_pos: int
) -> List[List[torch.Tensor]]:
    """Run the base model and return per-position, per-layer full-space residuals.

    Returns: residuals[pos][layer] = (D,) tensor.
    """
    raise NotImplementedError(
        "capture_base_residuals: requires running base_model.generate_with_cache "
        "with hooks on every layer. Implementation deferred until base loader is "
        "wired up (Q3)."
    )


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train(cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load base + inputs.
    base = load_compiled_model(cfg)
    token_ids, readout_pos, gt_token = load_circuit_inputs(cfg)

    # 2. Capture base residuals (frozen reference).
    base_residuals = capture_base_residuals(base, token_ids, readout_pos)

    # 3. Build wrapper.
    wrapper = CompressedTransformer(base, d_compressed=cfg.d_compressed).to(
        cfg.device, dtype=cfg.dtype
    )
    if hasattr(wrapper, "init_pca"):
        # Stack a representative subset of base residuals.
        snap = torch.stack([r for pos_res in base_residuals for r in pos_res], dim=0)
        wrapper.init_pca(snap.to(cfg.device).to(cfg.dtype))

    # 4. Optimizer.
    optimizer = torch.optim.AdamW(
        [wrapper.W],
        lr=cfg.lr_max,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    # 5. Loop.
    log = []
    for step in range(cfg.n_steps):
        # Linear LR decay
        frac = step / max(cfg.n_steps - 1, 1)
        lr = cfg.lr_max + frac * (cfg.lr_min - cfg.lr_max)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward
        x_c = wrapper.embed_token(token_ids[0])
        layer_caches = [{} for _ in range(len(base.attn))]
        compressed_residuals = []
        for pos, tid in enumerate(token_ids):
            if pos > 0:
                x_c = wrapper.embed_token(tid)
            x_c, layer_full = wrapper.forward_step_compressed(
                x_c, pos=pos, layer_caches=layer_caches,
                capture_layer_residuals=True,
            )
            compressed_residuals.append(layer_full)

        # Output logits at readout position
        compressed_logits = wrapper.head_logits(x_c)
        with torch.no_grad():
            base_logits = base.head(base_residuals[readout_pos][-1])

        L_out = compute_out_loss(compressed_logits, base_logits)
        L_layer = compute_layer_loss(
            [r for pos_lr in compressed_residuals for r in pos_lr],
            [r for pos_lr in base_residuals for r in pos_lr],
        )
        loss = L_out + cfg.lambda_layer * L_layer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0:
            log.append({"step": step, "loss": float(loss),
                        "L_out": float(L_out), "L_layer": float(L_layer),
                        "lr": lr})
            print(f"step {step:6d}  loss={float(loss):.4e}  "
                  f"L_out={float(L_out):.4e}  L_layer={float(L_layer):.4e}")

    # 6. Save.
    torch.save({
        "W": wrapper.W.detach().cpu(),
        "cfg": dataclasses.asdict(cfg),
        "log": log,
    }, out_dir / f"compressed_d{cfg.d_compressed}_{cfg.algorithm}_{cfg.circuit_id}.pt")


if __name__ == "__main__":
    print(__doc__)
    print("\nThis scaffold is currently BLOCKED on Q1–Q3 (see top of file).")
    print("Once those are answered, fill load_compiled_model and load_circuit_inputs,")
    print("then run: python train.py")
