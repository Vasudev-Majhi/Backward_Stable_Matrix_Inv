"""Interpretability experiments for Transformer 2 (readout / idea2).

Same four experiments as interp_t1.py but on the readout transformer.
Token sequence: start, (skip, v_init_j)*N, readout, <PRED>, halt (2N+4).

Run from inside ~/craft_release/craft/cadj/idea2/ so that the
craft-flavoured transformer-vm imports resolve.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

from parse import parse_netlist  # noqa: E402
from direct_tokenize import tokenize_direct  # noqa: E402
from transformer_vm.attention.standard_cache import StandardKVCache  # noqa: E402
from transformer_vm.model.transformer import add_position_encoding  # noqa: E402
from transformer_vm.model.weights import load_weights  # noqa: E402

DATASET = os.path.join(_REPO_ROOT, "dataset/circuit_dataset_rv.jsonl")
MODEL_DIR = _CLAUDE_FILES
OUT_DIR = os.path.join(_REPO_ROOT, "craft/paper_experiments/results")
os.makedirs(OUT_DIR, exist_ok=True)

CIRCUITS = ["CKT_0001", "CKT_0015", "CKT_0067"]
PREDICTED = "<PRED>"


def load_circuit(cid: str) -> dict:
    with open(DATASET) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                return c
    raise KeyError(cid)


def load_t2(cid: str):
    """Returns (model, tok_to_idx, fixed_toks, truth, build_load_s)."""
    c = load_circuit(cid)
    netlist = c["Netlist"]
    truth = float(c["Ground_Truth_Vout"])
    mp = os.path.join(MODEL_DIR, f"model_{cid}_idea2.bin")
    if not os.path.exists(mp):
        raise FileNotFoundError(mp)
    t0 = time.time()
    model, _, tok_to_idx = load_weights(mp)
    load_s = time.time() - t0
    model.eval()
    fixed_toks, _ = tokenize_direct(netlist, v_step=None, k_levels=None)
    return model, tok_to_idx, fixed_toks, truth, load_s


def run_inference(model, tok_to_idx, fixed_toks):
    """Run a single readout pass. Returns (pred_v_text, pred_v_value, infer_s)."""
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = StandardKVCache(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    pos = 0
    pred_text = "v_0"
    x = None
    t0 = time.time()
    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                # No forward pass for <PRED> — just decode argmax over v_* vocab
                # using the residual stream produced by the previous (readout) token.
                logits = model.head(x)
                best_score = -1e18
                for name, idx in tok_to_idx.items():
                    if name.startswith("v_"):
                        s = logits[idx].item()
                        if s > best_score:
                            best_score = s
                            pred_text = name
                # do not advance pos
            else:
                tok_idx = tok_to_idx[tok]
                x_in = model.tok.weight[tok_idx].clone()
                add_position_encoding(x_in, pos)
                x, _ = _forward_step_residual(model, cache, x_in, pos)
                pos += 1
    infer_s = time.time() - t0
    pred_v = float(pred_text.split("_")[1]) * 0.05
    return pred_text, pred_v, infer_s


def _forward_step_residual(model, cache, x_initial, pos):
    """Standard forward step, also returns per-layer residuals."""
    x = x_initial.clone()
    snaps = [x.clone().detach()]
    for layer_idx, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
        out = cache.layer_step(layer_idx, k, q, v)
        x = x + attn.out_proj(out)
        gate, val = ff_in(x).chunk(2, dim=-1)
        x = x + ff_out(F.relu(gate) * val)
        snaps.append(x.clone().detach())
    return x, snaps


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-1: Head ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_head_ablation(cid: str, log) -> list[dict]:
    log(f"[E-INTERP-1] {cid} ...")
    model, tok_to_idx, fixed_toks, truth, _ = load_t2(cid)
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    d_model = model.attn[0].embed_dim
    d_head = d_model // n_heads
    log(f"   n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}, d_head={d_head}")
    pred_text_ref, pred_v_ref, _ = run_inference(model, tok_to_idx, fixed_toks)
    log(f"   baseline pred={pred_text_ref} ({pred_v_ref:.4f}V), truth={truth:.4f}V")

    rows = []
    for layer_idx in range(n_layers):
        out_proj_w = model.attn[layer_idx].out_proj.weight
        original = out_proj_w.data.clone()
        for head in range(n_heads):
            cols = slice(head * d_head, (head + 1) * d_head)
            out_proj_w.data = original.clone()
            out_proj_w.data[:, cols] = 0.0
            try:
                pt, pv, _ = run_inference(model, tok_to_idx, fixed_toks)
                err_vs_ref = abs(pv - pred_v_ref)
                err_vs_truth = abs(pv - truth)
                changed = pt != pred_text_ref
            except Exception as e:
                pt = "ERR"; pv = float("nan")
                err_vs_ref = float("nan"); err_vs_truth = float("nan")
                changed = True
                log(f"   layer {layer_idx} head {head} FAILED: {e}")
            rows.append(dict(
                circuit=cid, transformer="T2", layer=layer_idx, head=head,
                pred_text=pt, pred_v=pv,
                err_vs_ref_v=err_vs_ref, err_vs_truth_v=err_vs_truth,
                pred_changed=int(bool(changed)),
            ))
        out_proj_w.data = original
    log(f"[E-INTERP-1] {cid} done ({len(rows)} ablations)")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-2: Residual probing
# ──────────────────────────────────────────────────────────────────────────────

def run_residual_probing(cid: str, log) -> list[dict]:
    """Capture residual stream after each layer at each token. Compute the
    L2 norm and the entry at the head-output projection slot."""
    log(f"[E-INTERP-2] {cid} ...")
    model, tok_to_idx, fixed_toks, truth, _ = load_t2(cid)
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    d_model = model.attn[0].embed_dim
    cache = StandardKVCache(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    rows = []
    pos = 0
    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                continue  # no forward pass for <PRED>
            tok_idx = tok_to_idx[tok]
            x = model.tok.weight[tok_idx].clone()
            add_position_encoding(x, pos)
            _, snaps = _forward_step_residual(model, cache, x, pos)
            for layer in range(len(snaps)):
                v = snaps[layer]
                rows.append(dict(
                    circuit=cid, transformer="T2", token=tok, pos=pos,
                    layer=layer,
                    l2_norm=float(v.norm().item()),
                    abs_max=float(v.abs().max().item()),
                    nz_count=int((v.abs() > 1e-12).sum().item()),
                ))
            pos += 1
    log(f"[E-INTERP-2] {cid} done ({len(rows)} rows)")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-3: Attention patterns
# ──────────────────────────────────────────────────────────────────────────────

def run_attention_patterns(cid: str, log) -> tuple[list[dict], dict]:
    log(f"[E-INTERP-3] {cid} ...")
    model, tok_to_idx, fixed_toks, truth, _ = load_t2(cid)
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    d_model = model.attn[0].embed_dim
    d_head = d_model // n_heads
    cache = StandardKVCache(n_layers, n_heads)
    K_cache: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    rows = []
    pos = 0
    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                continue  # no forward pass for <PRED>
            tok_idx = tok_to_idx[tok]
            x = model.tok.weight[tok_idx].clone()
            add_position_encoding(x, pos)

            for layer_idx, (attn, ff_in, ff_out) in enumerate(
                zip(model.attn, model.ff_in, model.ff_out, strict=True)
            ):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                K_cache[layer_idx].append(k.detach().clone())
                for head in range(n_heads):
                    q_h = q[head * d_head:(head + 1) * d_head]
                    scores = []
                    for past_k in K_cache[layer_idx]:
                        kh = past_k[head * d_head:(head + 1) * d_head]
                        scores.append(float((q_h * kh).sum().item()))
                    if scores:
                        am = int(np.argmax(scores))
                        rows.append(dict(
                            circuit=cid, transformer="T2", layer=layer_idx, head=head,
                            query_pos=pos, query_tok=tok,
                            argmax_key_pos=am, max_score=scores[am],
                            attn_entropy=_entropy(scores),
                            n_past_keys=len(scores),
                        ))
                out = cache.layer_step(layer_idx, k, q, v)
                x = x + attn.out_proj(out)
                gate, val = ff_in(x).chunk(2, dim=-1)
                x = x + ff_out(F.relu(gate) * val)
            pos += 1

    meta = dict(circuit=cid, n_layers=n_layers, n_heads=n_heads, tokens=fixed_toks)
    log(f"[E-INTERP-3] {cid} done ({len(rows)} rows)")
    return rows, meta


def _entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    s = np.array(scores, dtype=np.float64)
    s = s - s.max()
    p = np.exp(s)
    p = p / p.sum() + 1e-30
    return float(-np.sum(p * np.log(p)))


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-4: Compression
# ──────────────────────────────────────────────────────────────────────────────

def _quantize_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 64:
        return t
    out = t.clone()
    scale = float(out.abs().max().item())
    if scale == 0:
        return out
    levels = 2 ** (bits - 1) - 1
    q = (out / scale * levels).round().clamp(-levels, levels)
    return q / levels * scale


def _quantize_model(model, bits: int):
    for layer in range(len(model.attn)):
        a = model.attn[layer]
        a.in_proj_weight.data = _quantize_tensor(a.in_proj_weight.data, bits)
        a.out_proj.weight.data = _quantize_tensor(a.out_proj.weight.data, bits)
        model.ff_in[layer].weight.data = _quantize_tensor(model.ff_in[layer].weight.data, bits)
        model.ff_out[layer].weight.data = _quantize_tensor(model.ff_out[layer].weight.data, bits)
    model.tok.weight.data = _quantize_tensor(model.tok.weight.data, bits)


def _prune_model(model, frac: float):
    for layer in range(len(model.attn)):
        for w in (model.attn[layer].in_proj_weight,
                  model.attn[layer].out_proj.weight,
                  model.ff_in[layer].weight,
                  model.ff_out[layer].weight):
            if frac <= 0:
                continue
            a = w.data.abs().flatten().sort().values
            k = int(frac * a.numel())
            if k <= 0:
                continue
            thr = a[k - 1].item()
            mask = (w.data.abs() > thr).to(w.dtype)
            w.data.mul_(mask)


def run_compression(cid: str, log) -> list[dict]:
    log(f"[E-INTERP-4] {cid} ...")
    base_model, tok_to_idx, fixed_toks, truth, _ = load_t2(cid)
    pred_text_ref, pred_v_ref, t_ref = run_inference(base_model, tok_to_idx, fixed_toks)
    log(f"   baseline pred={pred_text_ref} ({pred_v_ref:.4f}V), truth={truth:.4f}V, infer={t_ref:.4f}s")

    settings = [
        ("float64", lambda m: None),
        ("int16", lambda m: _quantize_model(m, 16)),
        ("int8", lambda m: _quantize_model(m, 8)),
        ("int4", lambda m: _quantize_model(m, 4)),
        ("prune_10", lambda m: _prune_model(m, 0.1)),
        ("prune_50", lambda m: _prune_model(m, 0.5)),
        ("prune_90", lambda m: _prune_model(m, 0.9)),
    ]

    rows = []
    for name, mutator in settings:
        m_fresh, tok2, ftoks2, truth2, _ = load_t2(cid)
        mutator(m_fresh)
        try:
            pt, pv, t = run_inference(m_fresh, tok2, ftoks2)
            err_vs_ref = abs(pv - pred_v_ref)
            err_vs_truth = abs(pv - truth)
        except Exception as e:
            pt = "ERR"; pv = float("nan"); t = float("nan")
            err_vs_ref = float("nan"); err_vs_truth = float("nan")
            log(f"   {name} FAILED: {e}")
        rows.append(dict(
            circuit=cid, transformer="T2", setting=name,
            pred_text=pt, pred_v=pv,
            err_vs_ref_v=err_vs_ref, err_vs_truth_v=err_vs_truth,
            infer_s=t,
        ))
        log(f"   {name}: pred={pt}, err_vs_truth={err_vs_truth:.3e}V, infer={t:.4f}s")
    log(f"[E-INTERP-4] {cid} done")
    return rows


def write_csv(rows, path):
    if not rows:
        print(f"  (no rows for {path})")
        return
    keys = list({k for r in rows for k in r.keys()})
    front = [k for k in ("circuit", "transformer", "setting", "layer", "head",
                          "token", "pos", "slot", "value", "analytic", "abs_err",
                          "global_max_err", "rel_fro_err", "pred_text", "pred_v",
                          "err_vs_ref_v", "err_vs_truth_v", "max_err_vs_ref",
                          "max_err_vs_numpy", "infer_s",
                          "query_pos", "query_tok", "argmax_key_pos",
                          "max_score", "attn_entropy", "n_past_keys",
                          "l2_norm", "abs_max", "nz_count", "pred_changed")
             if k in keys]
    rest = sorted(k for k in keys if k not in front)
    keys = front + rest
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    log_path = os.path.join(OUT_DIR, "interp_t2.log")
    logf = open(log_path, "a")
    def log(msg):
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    log(f"=== interp_t2.py starting on circuits {CIRCUITS} ===")
    e1, e2, e3, e4 = [], [], [], []
    e3_meta = []
    for cid in CIRCUITS:
        try: e1.extend(run_head_ablation(cid, log))
        except Exception as e: log(f"E-INTERP-1 {cid} failed: {e}")
        try: e2.extend(run_residual_probing(cid, log))
        except Exception as e: log(f"E-INTERP-2 {cid} failed: {e}")
        try:
            r, m = run_attention_patterns(cid, log)
            e3.extend(r); e3_meta.append(m)
        except Exception as e: log(f"E-INTERP-3 {cid} failed: {e}")
        try: e4.extend(run_compression(cid, log))
        except Exception as e: log(f"E-INTERP-4 {cid} failed: {e}")
    write_csv(e1, os.path.join(OUT_DIR, "head_ablation_T2.csv"))
    write_csv(e2, os.path.join(OUT_DIR, "residual_probing_T2.csv"))
    write_csv(e3, os.path.join(OUT_DIR, "attention_patterns_T2.csv"))
    with open(os.path.join(OUT_DIR, "attention_patterns_T2_meta.json"), "w") as f:
        json.dump(e3_meta, f, indent=2)
    write_csv(e4, os.path.join(OUT_DIR, "compression_T2.csv"))
    log("=== interp_t2.py DONE ===")
    logf.close()


if __name__ == "__main__":
    main()
