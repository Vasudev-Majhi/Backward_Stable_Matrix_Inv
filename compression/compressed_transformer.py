"""
Tracr §5-style compression wrapper for the CRAFT VanillaTransformer.

Protocol (Lindner et al., NeurIPS 2023, §5):
  - Original residual stream lives in R^D where D = base.d_model.
  - Trainable projection W of shape (d, D) with d <= D.
  - Reads from residual: x_full = W^T @ x_compressed     (d -> D)
  - Writes to residual:   x_compressed += W @ delta_full (D -> d)
  - All other parameters frozen.

This wrapper implements one forward step for a single position. Training-time
loops over positions are in train.py.

OUTSTANDING DECISIONS (need user confirmation; see paper_data.md §6 T3):
  - which compiled model is the subject (LU-direct vs Jacobi vs iJacobi vs Direct-S)
  - which circuit(s) to compress
  - GPU vs CPU, float32 vs float64
  - cache behaviour during training (this scaffold disables the KV cache)
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_position_encoding(x: torch.Tensor, pos: int) -> None:
    """Mirror of transformer_vm.model.transformer.add_position_encoding (in-place)."""
    x[0] += pos
    x[1] += 1.0 / math.log(2) - 1.0 / math.log(pos + 2)
    x[2] += pos * pos


class CompressedTransformer(nn.Module):
    """Wraps a frozen VanillaTransformer with a trainable rank-d projection W.

    The compressed residual lives in R^d. At each layer we:
      1. decompress: x_full = W^T x_c                   (D-vector)
      2. compute attn / FFN updates against x_full as in the base model
      3. compress and add: x_c += W @ update_full

    The full-space residual at layer entry is faithful to the base model iff
    W^T W = I_D (i.e. d = D and W is orthogonal). For d < D, the compression is
    lossy and gradient descent must trade off output accuracy against capacity.
    """

    def __init__(self, base, d_compressed: int, init: str = "pca"):
        super().__init__()
        if not hasattr(base, "tok"):
            raise ValueError("base must be a VanillaTransformer-shaped module")
        self.base = base
        self.D = base.tok.embedding_dim
        self.d = int(d_compressed)
        if self.d < 1 or self.d > self.D:
            raise ValueError(f"d_compressed={d_compressed} not in [1, {self.D}]")

        # Freeze the base model.
        for p in self.base.parameters():
            p.requires_grad = False

        # Trainable W in R^(d x D).
        W = torch.empty(self.d, self.D)
        if init == "random":
            nn.init.orthogonal_(W)
        elif init == "pca":
            # Lazy: caller can call init_pca() once they have a residual snapshot.
            nn.init.orthogonal_(W)
        elif init == "identity_first_d":
            with torch.no_grad():
                W.zero_()
                for i in range(self.d):
                    W[i, i] = 1.0
        else:
            raise ValueError(f"unknown init '{init}'")
        self.W = nn.Parameter(W)

    # ---------------------------------------------------------------- I/O ---
    def expand(self, x_c: torch.Tensor) -> torch.Tensor:
        """Compressed -> full. (..., d) -> (..., D)."""
        return x_c @ self.W  # since W is (d, D), x_c @ W maps d -> D

    def compress(self, x_full: torch.Tensor) -> torch.Tensor:
        """Full -> compressed. (..., D) -> (..., d)."""
        return x_full @ self.W.T  # (D) @ (D, d) -> (d)

    # ----------------------------------------------- single-position step ---
    def forward_step_compressed(
        self,
        x_c: torch.Tensor,            # (d,) compressed residual
        pos: int,
        layer_caches: Optional[list] = None,
        capture_layer_residuals: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """One position through all layers. Returns (final x_c, per-layer x_full snapshots)."""
        # decompress current residual into the full space.
        x = self.expand(x_c).clone()  # (D,)

        # NOTE: position encoding is added in the *full* space (this matches the
        # base model where pos features land on dims 0,1,2). After the addition
        # we re-compress to update x_c, then continue. Equivalently, we can keep
        # operating in full space until the residual updates settle and only
        # compress at the end of each layer; we choose per-layer compression to
        # match Tracr §5 (one read+one write per layer).
        add_position_encoding(x, pos)
        x_c = self.compress(x)

        layer_residuals_full = []
        if capture_layer_residuals:
            layer_residuals_full.append(x.detach().clone())

        for layer_idx, (attn, ff_in, ff_out) in enumerate(
            zip(self.base.attn, self.base.ff_in, self.base.ff_out, strict=True)
        ):
            # READ (compressed -> full).
            x = self.expand(x_c)

            # ------- attention sublayer, NO KV CACHE (full attention) --------
            # During compression training we run full-sequence attention, not
            # streaming generation; the loop here is structured for a single
            # token because train.py drives the position loop. For self-
            # attention against a stored buffer of previous-position keys/values,
            # callers can pass layer_caches as a list of dicts {"K":, "V":}
            # appended each step.
            qkv = attn.in_proj_weight @ x  # (3*D,)
            q, k, v = qkv.chunk(3, dim=-1)

            if layer_caches is None:
                attn_out = attn.out_proj(v)  # degenerate: only the current token
            else:
                cache = layer_caches[layer_idx]
                cache.setdefault("K", []).append(k)
                cache.setdefault("V", []).append(v)
                K = torch.stack(cache["K"], dim=0)        # (T+1, D)
                V = torch.stack(cache["V"], dim=0)        # (T+1, D)
                # multi-head reshape
                H = attn.num_heads
                dh = self.D // H
                q_h = q.view(H, dh)
                K_h = K.view(-1, H, dh).transpose(0, 1)   # (H, T+1, dh)
                V_h = V.view(-1, H, dh).transpose(0, 1)   # (H, T+1, dh)
                scores = (q_h.unsqueeze(1) * K_h).sum(-1) / math.sqrt(dh)  # (H, T+1)
                w = torch.softmax(scores, dim=-1)          # (H, T+1)
                out_h = (w.unsqueeze(-1) * V_h).sum(dim=1)  # (H, dh)
                attn_out = attn.out_proj(out_h.reshape(self.D))

            # WRITE (full delta -> compressed update).
            x_c = x_c + self.compress(attn_out)

            # ----------------------------- FFN sublayer (ReGLU) -------------
            x = self.expand(x_c)
            gate, val = ff_in(x).chunk(2, dim=-1)
            ffn_out = ff_out(F.relu(gate) * val)
            x_c = x_c + self.compress(ffn_out)

            if capture_layer_residuals:
                layer_residuals_full.append(self.expand(x_c).detach().clone())

        return x_c, layer_residuals_full

    # ------------------------------------------------- vocab head readout ---
    def head_logits(self, x_c: torch.Tensor) -> torch.Tensor:
        """Apply the (frozen) vocab head to the decompressed residual."""
        x = self.expand(x_c)
        return self.base.head(x)

    # ------------------------------------------------------- token embed ---
    def embed_token(self, token_id: int) -> torch.Tensor:
        """Compressed embedding for a single token id."""
        x_full = self.base.tok.weight[token_id].clone()
        return self.compress(x_full)

    # ------------------------------------------------------------- utils ---
    @torch.no_grad()
    def init_pca(self, residual_snapshots: torch.Tensor) -> None:
        """Initialize W from PCA of observed residuals (recommended by Tracr §5).

        residual_snapshots: (M, D) tensor of full-space residuals collected
        from running the base model on a representative set of inputs.
        """
        if residual_snapshots.dim() != 2 or residual_snapshots.size(1) != self.D:
            raise ValueError(f"expected (M, {self.D}); got {residual_snapshots.shape}")
        # SVD: snapshots = U S V^T; top-d right singular vectors are the basis.
        snap = residual_snapshots - residual_snapshots.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(snap, full_matrices=False)
        self.W.data.copy_(Vt[: self.d])  # (d, D)
