"""Run a single circuit using the idea2 readout model — HullKVCache variant.

Drop-in replacement for _server_run_idea2.py that uses HullKVCache for O(log n)
attention instead of StandardKVCache. Everything else is identical.
"""
from __future__ import annotations

import os
import sys
import time

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl")
DATASET_PATH = os.environ.get("CRAFT_DATASET", _DEFAULT_DATASET)
MODEL_DIR    = _CLAUDE_FILES
PREDICTED    = "<PRED>"


def _load_circuit(cid: str) -> dict:
    import json
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                return c
    raise KeyError(f"Circuit {cid} not found")


def run_circuit_idea2(
    cid: str,
    tol: float = 0.05,
    v_step: int | None = None,
    k_levels: int | None = None,
    verbose: bool = False,
    model_dir: str = MODEL_DIR,
) -> tuple[str, float, float, float, int]:
    """Run idea2 readout model for cid using HullKVCache.

    Returns (status, pred_v, truth_v, infer_s, n_tokens).
    """
    import torch
    import torch.nn.functional as F
    from cadj_reference import V_STEP as DEFAULT_V_STEP
    from direct_tokenize import tokenize_direct
    from parse import parse_netlist
    from transformer_vm.attention.hull_cache import HullKVCache
    from transformer_vm.model.transformer import add_position_encoding
    from transformer_vm.model.weights import load_weights

    c       = _load_circuit(cid)
    netlist = c["Netlist"]
    target  = int(c["Target_Node"])
    truth   = float(c["Ground_Truth_Vout"])

    model_path = os.path.join(model_dir, f"model_{cid}_idea2.bin")
    if not os.path.exists(model_path):
        return "NO_MODEL", 0.0, truth, 0.0, 0

    t0 = time.time()

    model, _, tok_to_idx = load_weights(model_path)
    model.eval()

    fixed_toks, _ = tokenize_direct(netlist, v_step=v_step, k_levels=k_levels)
    n_tokens = len(fixed_toks)
    effective_v_step = DEFAULT_V_STEP if v_step is None else int(v_step)

    n_layers = len(model.attn)
    n_heads  = model.attn[0].num_heads
    cache    = HullKVCache(n_layers, n_heads)

    # Wire up tiebreak flags if the model has them
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    predicted_vs: list[str] = []
    x: torch.Tensor | None = None
    pos = 0

    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                logits = model.head(x)  # type: ignore[arg-type]
                best_name, best_score = "v_0", -1e18
                for name, idx in tok_to_idx.items():
                    if name.startswith("v_"):
                        s = logits[idx].item()
                        if s > best_score:
                            best_score, best_name = s, name
                predicted_vs.append(best_name)
                if verbose:
                    print(f"  pos {pos} PRED -> {best_name} (score {best_score:.1f})")
                tok_idx = tok_to_idx[best_name]
            else:
                tok_idx = tok_to_idx.get(tok, tok_to_idx.get("start", 0))

            emb = model.tok.weight[tok_idx].clone()
            add_position_encoding(emb, pos)
            x = emb

            for layer_idx, (attn, ff_in, ff_out) in enumerate(
                zip(model.attn, model.ff_in, model.ff_out, strict=True)
            ):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                out = cache.layer_step(layer_idx, k, q, v)
                x = x + attn.out_proj(out)
                gate, val = ff_in(x).chunk(2, dim=-1)
                x = x + ff_out(F.relu(gate) * val)

            pos += 1

    final_vk = predicted_vs[-1]
    k_val = int(final_vk.split("_")[1])
    pred_v = k_val * effective_v_step / 10000.0

    infer_s = time.time() - t0
    err = abs(pred_v - truth)
    status = "PASS" if err <= tol else "FAIL"
    return status, pred_v, truth, infer_s, n_tokens


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run idea2 circuit inference (HullKVCache)")
    ap.add_argument("circuit_id")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    status, pred_v, truth, infer_s, n_tokens = run_circuit_idea2(
        args.circuit_id, tol=args.tol, verbose=args.verbose, model_dir=args.model_dir,
    )
    err = abs(pred_v - truth)
    print(f"{status}  {args.circuit_id}  pred={pred_v:.4f}  truth={truth:.4f}  "
          f"err={err:.4f}  infer={infer_s:.3f}s  tokens={n_tokens}")


if __name__ == "__main__":
    main()
