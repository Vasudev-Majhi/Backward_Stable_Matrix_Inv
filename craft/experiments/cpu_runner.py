"""Real compiled-transformer inference on CPU.

Used for E2 spot-check (confirm DSL ≡ real transformer) and E10 (wall-clock
timing vs sequence length). Tries HullKVCache if available, else falls back
to StandardKVCache.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import os
import time

import torch
import torch.nn.functional as F

from interpreter import V_STEP
from tokenize_netlist import PREDICTED, tokenize


def _load_cache_class(prefer_hull: bool = True):
    """Return (cache_class, cache_name). Tries Hull if prefer_hull else Standard."""
    if prefer_hull:
        try:
            from transformer_vm.attention.hull_cache import HullKVCache
            # Trigger extension compilation on first use:
            _ = HullKVCache(1, 1)
            return HullKVCache, "hull"
        except Exception:  # noqa: BLE001
            pass
    from transformer_vm.attention.standard_cache import StandardKVCache
    return StandardKVCache, "standard"


def run_one_transformer(
    model_bin_path: str,
    netlist: str,
    target_node: int,
    T: int,
    *,
    cache: str = "auto",  # "auto" | "standard" | "hull"
) -> dict:
    """Run the compiled transformer on CPU.

    Returns:
      pred_V, final_vk, seq_length, infer_runtime_s, load_runtime_s, cache_used.
    """
    from transformer_vm.model.transformer import add_position_encoding
    from transformer_vm.model.weights import load_weights

    t0 = time.time()
    model, all_tokens, tok_to_idx = load_weights(model_bin_path)
    model.eval()
    load_runtime_s = time.time() - t0

    if cache == "auto":
        cache_class, cache_name = _load_cache_class(prefer_hull=True)
    elif cache == "hull":
        cache_class, cache_name = _load_cache_class(prefer_hull=True)
        if cache_name != "hull":
            raise RuntimeError("Hull cache requested but not available")
    else:
        from transformer_vm.attention.standard_cache import StandardKVCache
        cache_class, cache_name = StandardKVCache, "standard"

    fixed_toks, _, _ = tokenize(netlist, target_node, T=T)
    seq_length = len(fixed_toks)

    kv = cache_class(len(model.attn), model.attn[0].num_heads)

    t_infer = time.time()
    pos = 0
    x = None
    predicted_log: list[str] = []

    v_token_indices = [(n, tok_to_idx[n]) for n in tok_to_idx if n.startswith("v_")]

    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                logits = model.head(x)
                best_name = None
                best_score = -1e18
                for n, idx in v_token_indices:
                    s = logits[idx].item()
                    if s > best_score:
                        best_score = s
                        best_name = n
                if best_name is None:
                    best_name = "v_0"
                predicted_log.append(best_name)
                tok_idx = tok_to_idx[best_name]
            else:
                tok_idx = tok_to_idx.get(tok, tok_to_idx.get("start", 0))

            x = model.tok.weight[tok_idx].clone()
            add_position_encoding(x, pos)
            for li, (attn, ff_in, ff_out) in enumerate(
                zip(model.attn, model.ff_in, model.ff_out, strict=True)
            ):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                out = kv.layer_step(li, k, q, v)
                x = x + attn.out_proj(out)
                gate, val = ff_in(x).chunk(2, dim=-1)
                x = x + ff_out(F.relu(gate) * val)
            pos += 1

    infer_runtime_s = time.time() - t_infer

    final_vk = predicted_log[-1] if predicted_log else "v_0"
    if final_vk.startswith("v_"):
        k = int(final_vk.split("_")[1])
        pred_V = k * V_STEP / 10000.0
    else:
        pred_V = 0.0

    return {
        "pred_V": pred_V,
        "final_vk": final_vk,
        "seq_length": seq_length,
        "infer_runtime_s": infer_runtime_s,
        "load_runtime_s": load_runtime_s,
        "cache_used": cache_name,
    }


if __name__ == "__main__":
    import json
    import sys

    cid = sys.argv[1] if len(sys.argv) > 1 else "CKT_0001"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    with open("dataset/circuit_dataset_rv.jsonl") as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                break
        else:
            raise SystemExit(f"{cid} not found")

    mbin = os.path.join(os.path.dirname(__file__), "..", f"model_{cid}.bin")
    if not os.path.exists(mbin):
        raise SystemExit(f"Model not found: {mbin}. Run build.py first.")

    r = run_one_transformer(mbin, c["Netlist"], int(c["Target_Node"]), T)
    err = abs(r["pred_V"] - c["Ground_Truth_Vout"])
    status = "PASS" if err <= 0.05 else "FAIL"
    print(
        f"{status} {cid} T={T} pred={r['pred_V']:.4f}V "
        f"truth={c['Ground_Truth_Vout']:.4f}V err={err:.4f}V "
        f"seq={r['seq_length']} infer={r['infer_runtime_s']:.2f}s "
        f"cache={r['cache_used']}"
    )
