"""
Compression training — runs on the server.

Path A: discrete Jacobi compiled model (CKT_0001 first).

Layout assumed on server:
  ~/craft_release/craft/      <-- Jacobi runner + interpreter
  ~/craft_release/transformer-vm/     <-- model architecture
  ~/craft_release/dataset/            <-- circuit_dataset_rv.jsonl
  ~/craft_release/venv/bin/python     <-- venv with torch
  ~/craft_release/compression/        <-- this script + compressed_transformer.py

Usage on server:
    cd ~/craft_release/compression
    CRAFT_DATASET=~/craft_release/dataset/circuit_dataset_rv.jsonl \
      ~/craft_release/venv/bin/python server_train.py CKT_0001 \
        --d-compressed 16 --steps 200 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F


# Path setup: import craft (for tokenize_netlist, parse, jacobi_reference)
# and transformer-vm (for the model class + load_weights).
HOME = Path(os.path.expanduser("~"))
ROOT = HOME / "craft_release"
sys.path.insert(0, str(ROOT / "transformer-vm"))
sys.path.insert(0, str(ROOT / "craft"))

from transformer_vm.attention.standard_cache import StandardKVCache  # type: ignore
from transformer_vm.model.transformer import add_position_encoding  # type: ignore
from transformer_vm.model.weights import load_weights  # type: ignore

# craft imports
from interpreter import V_STEP  # type: ignore
from jacobi_reference import _auto_T  # type: ignore
from parse import parse_netlist  # type: ignore
from tokenize_netlist import tokenize  # type: ignore

PREDICTED = "<PRED>"
DATASET_PATH = os.environ.get(
    "CRAFT_DATASET", str(ROOT / "dataset" / "circuit_dataset_rv.jsonl")
)
MODEL_DIR = ROOT / "craft"

from compressed_transformer import CompressedTransformer  # local


# =========================================================================
# loaders
# =========================================================================
def load_compiled_model(circuit_id: str):
    mpath = MODEL_DIR / f"model_{circuit_id}.bin"
    if not mpath.exists():
        raise FileNotFoundError(mpath)
    model, all_tokens, tok_to_idx = load_weights(str(mpath))
    model.eval()
    return model, all_tokens, tok_to_idx


def load_circuit_inputs(circuit_id: str, T: int | None = None) -> Tuple[
    List[int], int, str, dict, int
]:
    """Returns (token_ids, readout_pos, truth_token_name, dataset_entry, target_node)."""
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == circuit_id:
                break
        else:
            raise KeyError(circuit_id)

    netlist = c["Netlist"]
    target = int(c["Target_Node"])
    truth = float(c["Ground_Truth_Vout"])

    pc = parse_netlist(netlist)
    if T is None:
        T = _auto_T(pc.num_nodes)
    fixed_toks, _, _ = tokenize(netlist, target, T=T)

    # the readout position is the position of the final <PRED> token
    readout_pos = max(i for i, t in enumerate(fixed_toks) if t == PREDICTED)
    return fixed_toks, readout_pos, truth, c, target


# =========================================================================
# capture base residuals — full-space layer outputs at every position
# =========================================================================
def _resolve_trajectory(model, fixed_toks: List[str], tok_to_idx) -> Tuple[List[int], int]:
    """Walk the sequence with the base model, replacing each <PRED> with the
    base-predicted v_ token id. Returns (token_id_list, final_readout_pos).

    Optimized: at <PRED> positions we argmax over the v_ token slice in a
    single GPU op (rather than 480 separate .item() calls, which would force
    a GPU sync per token and dominate runtime for long sequences)."""
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = StandardKVCache(n_layers, n_heads)

    # Precompute the (sorted) tensor of v_ token vocab indices and a parallel
    # array of their integer ids for O(1) gather + lookup.
    v_token_ids = sorted(idx for name, idx in tok_to_idx.items() if name.startswith("v_"))
    v_idx_t = torch.tensor(v_token_ids, dtype=torch.long, device=model.tok.weight.device)

    pos = 0
    last_x = None
    resolved: list[int] = []
    final_readout_pos = -1
    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                assert last_x is not None
                logits = model.head(last_x)  # (vocab,)
                v_logits = logits.index_select(0, v_idx_t)  # (n_v,)
                best_in_v = int(v_logits.argmax().item())   # ONE sync per <PRED>
                tok_idx = v_token_ids[best_in_v]
                final_readout_pos = pos
            else:
                if tok not in tok_to_idx:
                    tok = "start"
                tok_idx = tok_to_idx[tok]
            resolved.append(tok_idx)
            x = model.tok.weight[tok_idx].clone()
            add_position_encoding(x, pos)
            for li, (attn, ff_in, ff_out) in enumerate(
                zip(model.attn, model.ff_in, model.ff_out, strict=True)
            ):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                out = cache.layer_step(li, k, q, v)
                x = x + attn.out_proj(out)
                gate, val = ff_in(x).chunk(2, dim=-1)
                x = x + ff_out(F.relu(gate) * val)
            last_x = x
            pos += 1
    return resolved, final_readout_pos


def _batched_forward(
    model_or_wrapper, token_ids: torch.Tensor, dtype, capture_layer_outputs: bool,
    is_compressed: bool,
):
    """Sequence-batched forward over all T tokens with a causal mask.

    Works for both base model (is_compressed=False) and compressed wrapper
    (is_compressed=True). Returns a dict with:
      - layer_outputs (list of (T, D) full-space tensors per layer, including layer 0 input)
        if capture_layer_outputs=True
      - final_residual_full: (T, D) full-space residuals after all layers
    """
    import math
    if is_compressed:
        base = model_or_wrapper.base
        wrapper = model_or_wrapper
        D = wrapper.D
    else:
        base = model_or_wrapper
        wrapper = None
        D = base.tok.embedding_dim
    n_layers = len(base.attn)
    n_heads = base.attn[0].num_heads
    dh = D // n_heads
    T = token_ids.size(0)

    # Embed + position encode (full-space).
    X_full = base.tok.weight[token_ids].to(dtype).clone()  # (T, D)
    for pos in range(T):
        add_position_encoding(X_full[pos], pos)

    # Causal mask: True at positions to mask out (j > i).
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=X_full.device), diagonal=1,
    )

    if is_compressed:
        X_c = wrapper.compress(X_full)
        current = X_c
    else:
        current = X_full

    layer_outputs: list[torch.Tensor] = []
    if capture_layer_outputs:
        out_full = wrapper.expand(current) if is_compressed else current
        layer_outputs.append(out_full.detach())

    for li, (attn, ff_in, ff_out) in enumerate(
        zip(base.attn, base.ff_in, base.ff_out, strict=True)
    ):
        if is_compressed:
            X_full = wrapper.expand(current)
        else:
            X_full = current

        # Project Q, K, V for all positions.
        in_proj = attn.in_proj_weight.to(dtype)
        QKV = X_full @ in_proj.T  # (T, 3D)
        Q, K, V = QKV.chunk(3, dim=-1)
        Q = Q.view(T, n_heads, dh).transpose(0, 1)  # (H, T, dh)
        K = K.view(T, n_heads, dh).transpose(0, 1)
        V_ = V.view(T, n_heads, dh).transpose(0, 1)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)  # (H, T, T)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e30)
        w = torch.softmax(scores, dim=-1)
        attn_h = w @ V_  # (H, T, dh)
        attn_out = attn_h.transpose(0, 1).reshape(T, D)
        out_proj_w = attn.out_proj.weight.to(dtype)
        attn_out = attn_out @ out_proj_w.T  # (T, D)

        if is_compressed:
            current = current + wrapper.compress(attn_out)
            X_full = wrapper.expand(current)
        else:
            current = current + attn_out
            X_full = current

        ff_in_w = ff_in.weight.to(dtype)
        ff_out_w = ff_out.weight.to(dtype)
        gv = X_full @ ff_in_w.T  # (T, 2*d_ffn)
        gate, val = gv.chunk(2, dim=-1)
        ffn_out = (F.relu(gate) * val) @ ff_out_w.T  # (T, D)

        if is_compressed:
            current = current + wrapper.compress(ffn_out)
        else:
            current = current + ffn_out

        if capture_layer_outputs:
            out_full = wrapper.expand(current) if is_compressed else current
            layer_outputs.append(out_full.detach() if not is_compressed else out_full)

    final_full = wrapper.expand(current) if is_compressed else current
    return {
        "layer_outputs": layer_outputs if capture_layer_outputs else None,
        "final_residual_full": final_full,
    }


@torch.no_grad()
def base_forward(model, fixed_toks: List[int], tok_to_idx, capture: bool) -> dict:
    """Replicate runner.py's forward loop, optionally storing per-layer residuals.

    Returns dict with:
       layer_residuals[pos][layer_idx] = (D,) tensor (only if capture=True)
       final_x[pos] = (D,) tensor (after all layers at that pos)
       readout_logits = (vocab,) tensor at the final <PRED> position
       readout_pred_token_idx
    """
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = StandardKVCache(n_layers, n_heads)
    layer_residuals: list[list[torch.Tensor]] = []
    final_xs: list[torch.Tensor] = []

    pos = 0
    last_x = None
    last_predicted_logits = None
    pred_token_idx = -1

    for tok in fixed_toks:
        if tok == PREDICTED:
            # Score all v_ tokens at the previous position's hidden state.
            assert last_x is not None
            logits = model.head(last_x)
            last_predicted_logits = logits.detach().clone()
            best_name, best_score = None, -1e18
            for name, idx in tok_to_idx.items():
                if name.startswith("v_"):
                    s = logits[idx].item()
                    if s > best_score:
                        best_score, best_name = s, name
            pred_token_idx = tok_to_idx[best_name]
            tok_idx = pred_token_idx
        else:
            if tok not in tok_to_idx:
                tok = "start"
            tok_idx = tok_to_idx[tok]

        # Build x and run the layers.
        x = model.tok.weight[tok_idx].clone()
        add_position_encoding(x, pos)

        per_layer = []
        if capture:
            per_layer.append(x.detach().clone())  # layer-0 input

        for li, (attn, ff_in, ff_out) in enumerate(
            zip(model.attn, model.ff_in, model.ff_out, strict=True)
        ):
            q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
            out = cache.layer_step(li, k, q, v)
            x = x + attn.out_proj(out)
            gate, val = ff_in(x).chunk(2, dim=-1)
            x = x + ff_out(F.relu(gate) * val)
            if capture:
                per_layer.append(x.detach().clone())

        if capture:
            layer_residuals.append(per_layer)
        final_xs.append(x.detach().clone())
        last_x = x
        pos += 1

    return {
        "layer_residuals": layer_residuals if capture else None,
        "final_x": final_xs,
        "readout_logits": last_predicted_logits,
        "pred_token_idx": pred_token_idx,
    }


# =========================================================================
# compressed forward — mirrors base, but residual lives in d-dim
# =========================================================================
def compressed_forward(
    wrapper: CompressedTransformer,
    fixed_toks: List[str],
    tok_to_idx: dict,
    capture: bool,
):
    """Compressed-space replica of base_forward with full attention (no cache).

    Re-implements the per-layer recurrence so we can interleave W^T reads and
    W writes. We don't use a KV cache — instead we keep a list of past
    full-space K and V vectors per layer and compute attention from scratch
    each step. Slower (O(seq^2)) but gradient-friendly and avoids the
    StandardKVCache's in-place ops which would block autograd.
    """
    base = wrapper.base
    D = wrapper.D
    n_layers = len(base.attn)
    n_heads = base.attn[0].num_heads
    dh = D // n_heads

    # Per-layer K, V buffers (lists of full-space vectors).
    Ks = [[] for _ in range(n_layers)]
    Vs = [[] for _ in range(n_layers)]

    layer_residuals: list[list[torch.Tensor]] = []
    final_xs_full: list[torch.Tensor] = []
    last_x_full = None
    readout_logits = None
    pred_token_idx = -1

    pos = 0
    import math

    for tok in fixed_toks:
        if tok == PREDICTED:
            assert last_x_full is not None
            head_w = base.head.weight.to(wrapper.W.dtype)
            logits = head_w @ last_x_full
            readout_logits = logits
            # Pick best v_ token; fall back to argmax over all if v_ scores are degenerate.
            v_scores = []
            for name, idx in tok_to_idx.items():
                if name.startswith("v_"):
                    v_scores.append((logits[idx].item(), name))
            if v_scores:
                v_scores.sort(reverse=True)
                best_score, best_name = v_scores[0]
            else:
                best_name = next(iter(tok_to_idx.keys()))
            pred_token_idx = tok_to_idx[best_name]
            tok_idx = pred_token_idx
        else:
            if tok not in tok_to_idx:
                tok = "start"
            tok_idx = tok_to_idx[tok]

        # Initial full-space x (token embedding + position encoding), in W's dtype.
        x_full = base.tok.weight[tok_idx].to(wrapper.W.dtype).clone()
        add_position_encoding(x_full, pos)

        # Move into compressed.
        x_c = wrapper.compress(x_full)

        per_layer = [wrapper.expand(x_c).detach().clone()] if capture else None

        for li, (attn, ff_in, ff_out) in enumerate(
            zip(base.attn, base.ff_in, base.ff_out, strict=True)
        ):
            # Decompress for layer reads.
            x_full = wrapper.expand(x_c)

            qkv = (attn.in_proj_weight.to(wrapper.W.dtype) @ x_full)
            q, k, v = qkv.chunk(3, dim=-1)

            Ks[li].append(k)
            Vs[li].append(v)
            K = torch.stack(Ks[li], dim=0)   # (T+1, D)
            V = torch.stack(Vs[li], dim=0)
            q_h = q.view(n_heads, dh)
            K_h = K.view(-1, n_heads, dh).transpose(0, 1)   # (H, T+1, dh)
            V_h = V.view(-1, n_heads, dh).transpose(0, 1)
            scores = (q_h.unsqueeze(1) * K_h).sum(-1) / math.sqrt(dh)
            w = torch.softmax(scores, dim=-1)
            head_out = (w.unsqueeze(-1) * V_h).sum(dim=1).reshape(D)
            attn_out = attn.out_proj.weight.to(wrapper.W.dtype) @ head_out

            x_c = x_c + wrapper.compress(attn_out)

            x_full = wrapper.expand(x_c)
            ff_in_w = ff_in.weight.to(wrapper.W.dtype)
            ff_out_w = ff_out.weight.to(wrapper.W.dtype)
            gate_val = ff_in_w @ x_full
            gate, val = gate_val.chunk(2, dim=-1)
            ffn_out = ff_out_w @ (F.relu(gate) * val)

            x_c = x_c + wrapper.compress(ffn_out)

            if capture:
                per_layer.append(wrapper.expand(x_c).detach().clone())

        if capture:
            layer_residuals.append(per_layer)
        last_x_full = wrapper.expand(x_c)
        final_xs_full.append(last_x_full)
        pos += 1

    return {
        "layer_residuals": layer_residuals if capture else None,
        "final_x": final_xs_full,
        "readout_logits": readout_logits,
        "pred_token_idx": pred_token_idx,
    }


# =========================================================================
# loss
# =========================================================================
def compression_loss(base_state, compressed_state, lambda_layer: float):
    """L_out (cross-entropy vs base argmax) + lambda_layer * L_layer (1 - cosine).

    Both losses are scale-invariant — important because the base model has
    HARD_K=1e10 baked into its Q-weights, so raw activations are at 1e10
    scale and MSE losses are ill-conditioned.
    """
    base_logits = base_state["readout_logits"]
    comp_logits = compressed_state["readout_logits"]
    base_pred = int(base_state["pred_token_idx"])

    # L_out: CE on the full vocab, target = base model's predicted token.
    target = torch.tensor([base_pred], device=comp_logits.device, dtype=torch.long)
    # Scale logits down so cross_entropy doesn't overflow exp(1e10).
    # Subtracting max is what softmax does internally; we mirror it explicitly.
    scaled = comp_logits - comp_logits.max().detach()
    L_out = F.cross_entropy(scaled.unsqueeze(0), target)

    # L_layer: 1 - cosine similarity, averaged across layers and positions.
    # Skip layer 0 (just embedding+pos; identical to base by construction at PCA init).
    L_layer = torch.tensor(0.0, dtype=L_out.dtype, device=L_out.device)
    n = 0
    if (
        compressed_state["layer_residuals"] is not None
        and base_state["layer_residuals"] is not None
    ):
        for c_pos, b_pos in zip(
            compressed_state["layer_residuals"], base_state["layer_residuals"]
        ):
            for c_lr, b_lr in zip(c_pos[1:], b_pos[1:]):
                b_cast = b_lr.to(c_lr.dtype)
                cos = F.cosine_similarity(
                    c_lr.unsqueeze(0), b_cast.unsqueeze(0), dim=-1, eps=1e-8
                ).squeeze()
                L_layer = L_layer + (1.0 - cos)
                n += 1
    if n > 0:
        L_layer = L_layer / n
    return L_out + lambda_layer * L_layer, L_out, L_layer


# =========================================================================
# main
# =========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuit_id")
    p.add_argument("--d-compressed", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr-max", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--lambda-layer", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 1 short forward pass + 50 steps + brief report.")
    p.add_argument("--out-dir", default=str(Path.cwd() / "results"))
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--max-seq", type=int, default=None,
                   help="Truncate the resolved trajectory to first N tokens "
                        "(but keep the final readout). Reduces O(T^2) attention cost. "
                        "For CKT_0001 with T=1000 iterations the natural sequence has "
                        "6010 tokens; 500 captures init + ~80 iterations + readout.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device={args.device} smoke={args.smoke}", flush=True)

    # ---- load -------------------------------------------------------------
    t0 = time.time()
    model, all_tokens, tok_to_idx = load_compiled_model(args.circuit_id)
    fixed_toks, readout_pos, truth, ds_entry, target = load_circuit_inputs(args.circuit_id, args.T)
    D = model.tok.embedding_dim
    n_layers = len(model.attn)
    print(f"[info] loaded model: D={D}, n_layers={n_layers}, vocab={len(all_tokens)}", flush=True)
    print(f"[info] sequence length: {len(fixed_toks)} tokens; readout at pos {readout_pos}", flush=True)
    print(f"[info] truth = {truth:.4f}V", flush=True)
    print(f"[info] load time {time.time()-t0:.2f}s", flush=True)

    # Move base model to target device BEFORE trajectory resolution. Without
    # this the resolution loop runs on CPU which is the dominant cost for long
    # sequences (6000+ tokens).
    t0 = time.time()
    model = model.to(args.device, dtype=torch.float64)
    print(f"[info] base model -> {args.device}: {time.time()-t0:.2f}s", flush=True)

    # Truncate fixed_toks BEFORE trajectory resolution. The trajectory resolver
    # walks token-by-token (cache.layer_step appends growing K,V lists; softmax
    # per step). For T=6010 this is 30k cache calls and dominates total time
    # even on GPU. We only need to capture base behavior on a prefix that
    # includes the algorithm structure + at least one <PRED>.
    if args.max_seq is not None and len(fixed_toks) > args.max_seq:
        # Find the latest <PRED> at or before max_seq; truncate to include it.
        keep_until = -1
        for i, t in enumerate(fixed_toks[: args.max_seq]):
            if t == PREDICTED:
                keep_until = i
        if keep_until < 0:
            # No <PRED> in the first max_seq tokens; just truncate.
            keep_until = args.max_seq - 1
        fixed_toks = fixed_toks[: keep_until + 1]
        print(f"[info] fixed_toks truncated to {len(fixed_toks)} tokens "
              f"(last <PRED> at pos {keep_until})", flush=True)

    # ---- shorten sequence in smoke mode ----------------------------------
    if args.smoke:
        # Keep tokens up to first <PRED> for a fast smoke test.
        first_pred = next(i for i, t in enumerate(fixed_toks) if t == PREDICTED)
        fixed_toks = fixed_toks[: first_pred + 1]
        readout_pos = first_pred
        print(f"[info] smoke: truncated to {len(fixed_toks)} tokens", flush=True)

    # ---- resolve base trajectory + capture residuals (BATCHED) -----------
    t0 = time.time()
    resolved_ids, final_readout_pos = _resolve_trajectory(model, fixed_toks, tok_to_idx)
    print(f"[info] base trajectory resolved: {time.time()-t0:.2f}s; "
          f"sequence length {len(resolved_ids)}; final readout at pos {final_readout_pos}",
          flush=True)
    # (truncation already applied to fixed_toks above; resolved_ids matches.)
    final_pred_token_id = resolved_ids[final_readout_pos]
    pred_name = next((n for n, i in tok_to_idx.items() if i == final_pred_token_id), "?")
    print(f"[info] base final predicted: {pred_name}", flush=True)

    token_ids_t = torch.tensor(resolved_ids, dtype=torch.long, device=args.device)
    t0 = time.time()
    base_state_b = _batched_forward(
        model, token_ids_t, dtype=torch.float64,
        capture_layer_outputs=True, is_compressed=False,
    )
    print(f"[info] base batched forward: {time.time()-t0:.2f}s; "
          f"layer_outputs len={len(base_state_b['layer_outputs'])}, "
          f"each shape {tuple(base_state_b['layer_outputs'][0].shape)}")
    base_layer_outs = [t.detach() for t in base_state_b["layer_outputs"]]
    base_final_full = base_state_b["final_residual_full"].detach()

    # The "readout" is the head applied to the residual at position
    # final_readout_pos - 1 (the position before <PRED>; the runner reads logits
    # from last_x which is the residual at the previous position).
    readout_idx = max(0, final_readout_pos - 1)
    base_readout_x = base_final_full[readout_idx]
    head_w = model.head.weight.to(torch.float64)
    base_readout_logits = head_w @ base_readout_x
    base_pred_idx = int(base_readout_logits.argmax().item())

    # ---- build wrapper ----------------------------------------------------
    # NOTE: keep everything in float64. Base model has HARD_K=1e10 baked into
    # Q-weights (saturation of softmax to argmax). float32 overflows the
    # softmax and produces NaN logits. Tracr's float32 default does not apply.
    wrapper = CompressedTransformer(model, d_compressed=args.d_compressed, init="random")
    wrapper = wrapper.to(args.device, dtype=torch.float64)
    wrapper.base.to(args.device)

    # PCA init from observed batched full-space residuals.
    snaps = torch.cat([r.reshape(-1, model.tok.embedding_dim) for r in base_layer_outs], dim=0)
    wrapper.init_pca(snaps.to(args.device).to(torch.float64))
    print(f"[info] PCA init from {snaps.size(0)} residuals; W shape {tuple(wrapper.W.shape)}", flush=True)

    # ---- training loop ----------------------------------------------------
    optimizer = torch.optim.AdamW(
        [wrapper.W], lr=args.lr_max, betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    log = []
    for step in range(args.steps):
        frac = step / max(args.steps - 1, 1)
        lr = args.lr_max + frac * (args.lr_min - args.lr_max)
        for g in optimizer.param_groups:
            g["lr"] = lr

        comp_state = _batched_forward(
            wrapper, token_ids_t, dtype=torch.float64,
            capture_layer_outputs=True, is_compressed=True,
        )
        # Readout logits at readout_idx
        comp_readout_x = comp_state["final_residual_full"][readout_idx]
        comp_logits = head_w @ comp_readout_x

        # L_out: MSE between softmax(comp_logits/T) and softmax(base_logits/T).
        # Bounded in [0, 1]; smooth gradients. The temperature absorbs the
        # HARD_K=1e10 baked into the base Q-weights; softmax saturation otherwise
        # makes plain CE ill-conditioned.
        with torch.no_grad():
            base_p = F.softmax(base_readout_logits.detach() / 1e10, dim=-1)
        comp_p = F.softmax(comp_logits / 1e10, dim=-1)
        L_out = F.mse_loss(comp_p, base_p)

        # L_layer: 1 - cosine similarity, averaged across (layer >= 1, position).
        L_layer = torch.tensor(0.0, dtype=torch.float64, device=args.device)
        n = 0
        for c_lo, b_lo in zip(comp_state["layer_outputs"][1:], base_layer_outs[1:]):
            cos = F.cosine_similarity(c_lo, b_lo, dim=-1, eps=1e-8)  # (T,)
            L_layer = L_layer + (1.0 - cos).mean()
            n += 1
        if n > 0:
            L_layer = L_layer / n

        loss = L_out + args.lambda_layer * L_layer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # also track an interpretable metric: argmax-match
        with torch.no_grad():
            comp_pred = int(comp_logits.argmax().item())
            argmax_match = int(comp_pred == base_pred_idx)

        if step % max(1, args.steps // 50) == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss),
                        "L_out": float(L_out), "L_layer": float(L_layer),
                        "argmax_match": argmax_match, "lr": lr})
            print(f"step {step:5d}  loss={float(loss):.4e}  "
                  f"L_out={float(L_out):.4e}  L_layer={float(L_layer):.4e}  "
                  f"match={argmax_match}  lr={lr:.2e}")

    # ---- save -------------------------------------------------------------
    out_path = out_dir / f"compressed_d{args.d_compressed}_jacobi_{args.circuit_id}.pt"
    torch.save({
        "W": wrapper.W.detach().cpu(),
        "log": log,
        "args": vars(args),
        "base_pred_token_idx": base_pred_idx,
    }, out_path)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
