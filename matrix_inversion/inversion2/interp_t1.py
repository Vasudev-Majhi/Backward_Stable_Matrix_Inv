"""Interpretability experiments for Transformer 1 (LU inversion).

Runs four experiments on a small set of circuits:
  E-INTERP-1: Per-head ablation attribution
  E-INTERP-2: Residual stream probing (linear-decoding intermediate state)
  E-INTERP-3: Attention pattern extraction
  E-INTERP-4: Weight-compression robustness

For each circuit we use the pre-built LU inversion model at
  ~/craft_release/craft/cadj/idea2/lu_models/CKT_XXXX/model_<hash>_lu.bin

Outputs go to ~/craft_release/craft/paper_experiments/results/
with filenames suffixed _T1.

Run from inside ~/craft_release/matrix_inversion/inversion2/ so that
inversion2-flavoured transformer-vm imports resolve (its _bootstrap.py is
loaded automatically when import runner_lu).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch

# Bootstrap: this script lives in inversion2/, so importing _bootstrap sets up paths
import _bootstrap  # noqa: F401

import runner_lu  # provides _forward_step, solve_column_lu, etc.
from transformer_vm.model.transformer import add_position_encoding
from transformer_vm.model.weights import load_weights

ROOT = os.path.expanduser("~/craft_release")
LU_MODEL_ROOT = os.path.join(ROOT, "craft/cadj/idea2/lu_models")
DATASET = os.path.join(ROOT, "dataset/circuit_dataset_rv.jsonl")
OUT_DIR = os.path.join(ROOT, "craft/paper_experiments/results")
os.makedirs(OUT_DIR, exist_ok=True)

CIRCUITS = ["CKT_0001", "CKT_0015", "CKT_0067"]


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_circuit(cid: str):
    sys.path.insert(0, os.path.join(ROOT, "craft"))
    from parse import parse_netlist
    with open(DATASET) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                pc = parse_netlist(c["Netlist"])
                # Build A_FF
                N = pc.num_nodes
                A = np.zeros((N, N), dtype=np.float64)
                for a, b, ohms in pc.resistors:
                    g = 1.0 / ohms
                    A[a, a] += g
                    A[b, b] += g
                    A[a, b] -= g
                    A[b, a] -= g
                free = [i for i, fx in enumerate(pc.is_fixed) if not fx]
                A_FF = A[np.ix_(free, free)]
                return c, pc, A_FF, free
    raise KeyError(cid)


def find_lu_model(cid: str) -> str:
    cand = glob.glob(os.path.join(LU_MODEL_ROOT, cid, "model_*_lu.bin"))
    if not cand:
        raise FileNotFoundError(f"No LU model for {cid} under {LU_MODEL_ROOT}")
    return cand[0]


def load_lu(cid: str):
    """Return (model, tok_to_idx, sidecar, A_FF, build_time_recorded)."""
    mp = find_lu_model(cid)
    sp = mp + ".slots.json"
    sidecar = json.load(open(sp))
    t0 = time.time()
    model, _, tok_to_idx = load_weights(mp)
    load_s = time.time() - t0
    model.eval()
    _, _, A_FF, _ = parse_circuit(cid)
    return model, tok_to_idx, sidecar, A_FF, load_s


def invert_lu(model, tok_to_idx, sidecar, A: np.ndarray) -> tuple[np.ndarray, float]:
    """Run full inversion. Returns (X, infer_time_seconds)."""
    n = A.shape[0]
    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])
    X = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        X[:, j] = runner_lu.solve_column_lu(
            model, tok_to_idx, n,
            slot_x_value, slot_b_value, slot_y_input, slot_x_new,
            b=e_j, verbose=False,
        )
    return X, time.time() - t0


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-1: Head ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_head_ablation(cid: str, log) -> list[dict]:
    """For each (layer, head): zero its slice of attn.out_proj and re-invert.

    Records per-column max error vs the un-ablated model.
    """
    log(f"[E-INTERP-1] {cid} ...")
    model, tok_to_idx, sidecar, A_FF, _ = load_lu(cid)
    n = A_FF.shape[0]
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    d_model = model.attn[0].embed_dim
    d_head = d_model // n_heads
    log(f"   n={n}, n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}, d_head={d_head}")

    # baseline
    X_ref, _ = invert_lu(model, tok_to_idx, sidecar, A_FF)
    X_truth = np.linalg.inv(A_FF)
    log(f"   baseline max|X - numpy|: {float(np.max(np.abs(X_ref - X_truth))):.2e}")

    results = []
    for layer_idx in range(n_layers):
        out_proj_w = model.attn[layer_idx].out_proj.weight
        original = out_proj_w.data.clone()
        for head in range(n_heads):
            cols = slice(head * d_head, (head + 1) * d_head)
            out_proj_w.data = original.clone()
            out_proj_w.data[:, cols] = 0.0
            try:
                X_ablated, _ = invert_lu(model, tok_to_idx, sidecar, A_FF)
                per_col_err = [float(np.max(np.abs(X_ablated[:, j] - X_ref[:, j])))
                               for j in range(n)]
                global_err = max(per_col_err)
                rel = float(np.linalg.norm(X_ablated - X_ref) / (np.linalg.norm(X_ref) + 1e-30))
            except Exception as e:
                per_col_err = [float("nan")] * n
                global_err = float("nan")
                rel = float("nan")
                log(f"   layer {layer_idx} head {head} FAILED: {e}")
            row = dict(
                circuit=cid, transformer="T1", layer=layer_idx, head=head,
                global_max_err=global_err, rel_fro_err=rel,
            )
            for j, e in enumerate(per_col_err):
                row[f"col_{j}_err"] = e
            results.append(row)
        # restore
        out_proj_w.data = original
    log(f"[E-INTERP-1] {cid} done ({len(results)} ablations)")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-2: Residual stream probing
# ──────────────────────────────────────────────────────────────────────────────

def _forward_with_residual(model, cache, x_initial, pos, n_layers):
    """Like _forward_step, but returns the residual stream after each layer."""
    import torch.nn.functional as F
    x = x_initial.clone()
    snapshots = [x.clone().detach()]  # before any layer (= layer 0 input)
    for layer_idx, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
        out = cache.layer_step(layer_idx, k, q, v)
        x = x + attn.out_proj(out)
        gate, val = ff_in(x).chunk(2, dim=-1)
        x = x + ff_out(F.relu(gate) * val)
        snapshots.append(x.clone().detach())
    return x, snapshots


def run_residual_probing(cid: str, log) -> list[dict]:
    """Capture residual stream after each layer at each token; check whether
    the algorithm-defined slot (x_new) matches the analytical y_i / x_i.

    For T1 this is direct: slot_x_new should hold y_i after fwd_i and x_i
    after bck_i. We measure the per-layer absolute error vs the analytic value.
    """
    log(f"[E-INTERP-2] {cid} ...")
    model, tok_to_idx, sidecar, A_FF, _ = load_lu(cid)
    n = A_FF.shape[0]
    n_layers = len(model.attn)
    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])

    # Reference y, x from numpy LU
    from scipy.linalg import lu  # try scipy; fall back to manual
    try:
        Pmat, L, U = lu(A_FF)
    except Exception:
        # Doolittle: simple manual LU
        L = np.eye(n)
        U = A_FF.copy()
        for i in range(n):
            for k in range(i + 1, n):
                if U[i, i] == 0:
                    raise RuntimeError("zero pivot")
                L[k, i] = U[k, i] / U[i, i]
                U[k, :] -= L[k, i] * U[i, :]

    # For column j: y = L^{-1} e_j, then x = U^{-1} y. We probe j=0.
    j = 0
    e_j = np.zeros(n); e_j[j] = 1.0
    # forward sub
    y_true = np.linalg.solve(L, e_j)
    x_true = np.linalg.solve(U, y_true)
    log(f"   y_true={y_true.tolist()}")
    log(f"   x_true={x_true.tolist()}")

    # Now run model token-by-token and record residual stream after each layer.
    # We must use the same slot patching as the runner.
    from transformer_vm.attention.standard_cache import StandardKVCache
    n_heads = model.attn[0].num_heads
    cache = StandardKVCache(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    tokens = (
        ["start"]
        + [f"init_{i}" for i in range(n)]
        + [f"fwd_{i}" for i in range(n)]
        + [f"bck_{i}" for i in range(n - 1, -1, -1)]
        + ["halt"]
    )

    rows = []
    pos = 0
    y_buf = [0.0] * n
    x_result = [0.0] * n
    with torch.no_grad():
        for tok in tokens:
            tok_idx = tok_to_idx[tok]
            x_init = model.tok.weight[tok_idx].clone()
            add_position_encoding(x_init, pos)

            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_b_value] = x_init[slot_b_value] + float(e_j[i])
                _, snaps = _forward_with_residual(model, cache, x_init, pos, n_layers)
                # record per-layer slot_x_new
                for layer in range(len(snaps)):
                    val = float(snaps[layer][slot_x_new].item())
                    rows.append(dict(
                        circuit=cid, transformer="T1", token=tok, pos=pos,
                        layer=layer, slot="x_new", value=val,
                        analytic=float(y_true[i]),
                        abs_err=abs(val - float(y_true[i])),
                    ))
                y_buf[i] = float(snaps[-1][slot_x_new].item())
                runner_lu._vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value, slot_x_new=slot_x_new,
                    new_value=y_buf[i],
                    extra={slot_b_value: float(e_j[i])},
                )

            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_y_input] = x_init[slot_y_input] + float(y_buf[i])
                _, snaps = _forward_with_residual(model, cache, x_init, pos, n_layers)
                for layer in range(len(snaps)):
                    val = float(snaps[layer][slot_x_new].item())
                    rows.append(dict(
                        circuit=cid, transformer="T1", token=tok, pos=pos,
                        layer=layer, slot="x_new", value=val,
                        analytic=float(x_true[i]),
                        abs_err=abs(val - float(x_true[i])),
                    ))
                x_result[i] = float(snaps[-1][slot_x_new].item())
                runner_lu._vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value, slot_x_new=slot_x_new,
                    new_value=x_result[i],
                    extra={slot_y_input: float(y_buf[i])},
                )
            else:
                _, _ = _forward_with_residual(model, cache, x_init, pos, n_layers)
            pos += 1
    log(f"[E-INTERP-2] {cid} done ({len(rows)} rows)")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-3: Attention pattern extraction
# ──────────────────────────────────────────────────────────────────────────────

def run_attention_patterns(cid: str, log) -> tuple[list[dict], dict]:
    """For each (layer, head, fwd_i token), record which past position carried
    the maximum attention score from the current query.

    Q, K computed as (in_proj_weight @ x).chunk(3) and the attention score is
    Q @ K^T (per-head). With HARD_K saturation the argmax is essentially
    one-hot, so we record the argmax position for each query.
    """
    log(f"[E-INTERP-3] {cid} ...")
    model, tok_to_idx, sidecar, A_FF, _ = load_lu(cid)
    n = A_FF.shape[0]
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    d_model = model.attn[0].embed_dim
    d_head = d_model // n_heads
    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])

    tokens = (
        ["start"]
        + [f"init_{i}" for i in range(n)]
        + [f"fwd_{i}" for i in range(n)]
        + [f"bck_{i}" for i in range(n - 1, -1, -1)]
        + ["halt"]
    )

    # Replay forward but capture (Q, K) at every layer & every position.
    from transformer_vm.attention.standard_cache import StandardKVCache
    import torch.nn.functional as F
    cache = StandardKVCache(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    # K_cache[layer] = list of K vectors per past position
    K_cache: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    rows = []
    pos = 0
    y_buf = [0.0] * n
    x_result = [0.0] * n
    e_j = np.zeros(n); e_j[0] = 1.0  # probe column 0

    # full attention matrix (n_layers, n_heads, T, T) but we'll just save
    # argmax per query because HARD_K makes it ~one-hot
    arg_attn = []  # list of dicts: layer, head, query_pos, query_tok, key_pos, max_score

    with torch.no_grad():
        for tok in tokens:
            tok_idx = tok_to_idx[tok]
            x_init = model.tok.weight[tok_idx].clone()
            add_position_encoding(x_init, pos)
            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_b_value] = x_init[slot_b_value] + float(e_j[i])

            x = x_init.clone()
            for layer_idx, (attn, ff_in, ff_out) in enumerate(
                zip(model.attn, model.ff_in, model.ff_out, strict=True)
            ):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                K_cache[layer_idx].append(k.detach().clone())
                # per-head attention score = Q_h . K_h
                for head in range(n_heads):
                    q_h = q[head * d_head:(head + 1) * d_head]
                    scores = []
                    for past_idx, past_k in enumerate(K_cache[layer_idx]):
                        kh = past_k[head * d_head:(head + 1) * d_head]
                        scores.append(float((q_h * kh).sum().item()))
                    if scores:
                        am = int(np.argmax(scores))
                        arg_attn.append(dict(
                            circuit=cid, transformer="T1", layer=layer_idx, head=head,
                            query_pos=pos, query_tok=tok,
                            argmax_key_pos=am, max_score=scores[am],
                            attn_entropy=float(_entropy(scores)),
                            n_past_keys=len(scores),
                        ))
                # forward step (use cache.layer_step for actual computation)
                out = cache.layer_step(layer_idx, k, q, v)
                x = x + attn.out_proj(out)
                gate, val = ff_in(x).chunk(2, dim=-1)
                x = x + ff_out(F.relu(gate) * val)

            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                y_buf[i] = float(x[slot_x_new].item())
                runner_lu._vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value, slot_x_new=slot_x_new,
                    new_value=y_buf[i],
                    extra={slot_b_value: float(e_j[i])},
                )
                # also patch our K_cache shadow with the new K
                new_kqv = model.attn[0].in_proj_weight @ x_init  # only layer 0 for shadow
                # actually re-compute K for all layers using x_init clone with slot patches
                x_corrected = x_init.clone()
                x_corrected[slot_x_value] = float(y_buf[i])
                x_corrected[slot_b_value] = float(e_j[i])
                x_corrected[slot_x_new] = float(y_buf[i])
                for li in range(n_layers):
                    nk = (model.attn[li].in_proj_weight @ x_corrected).chunk(3, dim=-1)[1]
                    K_cache[li][-1] = nk.detach().clone()
            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                # use analytic y for the patch (model's value should match)
                x_init[slot_y_input] = x_init[slot_y_input] + float(y_buf[i])
                x_result[i] = float(x[slot_x_new].item())
                runner_lu._vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value, slot_x_new=slot_x_new,
                    new_value=x_result[i],
                    extra={slot_y_input: float(y_buf[i])},
                )
                x_corrected = x_init.clone()
                x_corrected[slot_x_value] = float(x_result[i])
                x_corrected[slot_y_input] = float(y_buf[i])
                x_corrected[slot_x_new] = float(x_result[i])
                for li in range(n_layers):
                    nk = (model.attn[li].in_proj_weight @ x_corrected).chunk(3, dim=-1)[1]
                    K_cache[li][-1] = nk.detach().clone()
            pos += 1

    # Token labels for plotting
    meta = dict(circuit=cid, n=n, n_layers=n_layers, n_heads=n_heads,
                tokens=tokens)
    log(f"[E-INTERP-3] {cid} done ({len(arg_attn)} attention argmax records)")
    return arg_attn, meta


def _entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    s = np.array(scores, dtype=np.float64)
    s = s - s.max()
    p = np.exp(s)
    p = p / p.sum()
    p = p + 1e-30
    return float(-np.sum(p * np.log(p)))


# ──────────────────────────────────────────────────────────────────────────────
# E-INTERP-4: Weight-compression robustness
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
    """Magnitude-pruning: zero out smallest |w| entries by global threshold per tensor."""
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
    rows = []
    base_model, tok_to_idx, sidecar, A_FF, load_s = load_lu(cid)
    X_truth = np.linalg.inv(A_FF)
    X_ref, t_ref = invert_lu(base_model, tok_to_idx, sidecar, A_FF)
    base_err = float(np.max(np.abs(X_ref - X_truth)))
    log(f"   baseline max|X - numpy|={base_err:.3e}, infer={t_ref:.2f}s")

    settings = [
        ("float64", lambda m: None),
        ("int16", lambda m: _quantize_model(m, 16)),
        ("int8", lambda m: _quantize_model(m, 8)),
        ("int4", lambda m: _quantize_model(m, 4)),
        ("prune_10", lambda m: _prune_model(m, 0.1)),
        ("prune_50", lambda m: _prune_model(m, 0.5)),
        ("prune_90", lambda m: _prune_model(m, 0.9)),
    ]

    for name, mutator in settings:
        # reload fresh model to avoid compounding mutations
        model_fresh, tok_to_idx_f, sidecar_f, _, _ = load_lu(cid)
        mutator(model_fresh)
        try:
            t0 = time.time()
            X_c, _ = invert_lu(model_fresh, tok_to_idx_f, sidecar_f, A_FF)
            t = time.time() - t0
            err_vs_ref = float(np.max(np.abs(X_c - X_ref)))
            err_vs_truth = float(np.max(np.abs(X_c - X_truth)))
            rel = float(np.linalg.norm(X_c - X_ref) / (np.linalg.norm(X_ref) + 1e-30))
        except Exception as e:
            err_vs_ref = float("nan")
            err_vs_truth = float("nan")
            rel = float("nan")
            t = float("nan")
            log(f"   {name} FAILED: {e}")
        rows.append(dict(
            circuit=cid, transformer="T1", setting=name,
            max_err_vs_ref=err_vs_ref, max_err_vs_numpy=err_vs_truth,
            rel_fro_err=rel, infer_s=t,
        ))
        log(f"   {name}: max_err_vs_numpy={err_vs_truth:.3e}, infer={t:.2f}s")
    log(f"[E-INTERP-4] {cid} done")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: str):
    if not rows:
        print(f"  (no rows for {path})")
        return
    keys = list({k for r in rows for k in r.keys()})
    # stable order: known keys first, then extras
    front = [k for k in ("circuit", "transformer", "setting", "layer", "head",
                          "token", "pos", "slot", "value", "analytic", "abs_err",
                          "global_max_err", "rel_fro_err",
                          "max_err_vs_ref", "max_err_vs_numpy", "infer_s",
                          "query_pos", "query_tok", "argmax_key_pos",
                          "max_score", "attn_entropy", "n_past_keys")
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
    log_path = os.path.join(OUT_DIR, "interp_t1.log")
    logf = open(log_path, "a")
    def log(msg):
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== interp_t1.py starting on circuits {CIRCUITS} ===")

    e1_rows, e2_rows, e3_rows, e4_rows = [], [], [], []
    e3_meta = []

    for cid in CIRCUITS:
        try:
            e1_rows.extend(run_head_ablation(cid, log))
        except Exception as e:
            log(f"E-INTERP-1 {cid} failed: {e}")
        try:
            e2_rows.extend(run_residual_probing(cid, log))
        except Exception as e:
            log(f"E-INTERP-2 {cid} failed: {e}")
        try:
            rows3, meta3 = run_attention_patterns(cid, log)
            e3_rows.extend(rows3)
            e3_meta.append(meta3)
        except Exception as e:
            log(f"E-INTERP-3 {cid} failed: {e}")
        try:
            e4_rows.extend(run_compression(cid, log))
        except Exception as e:
            log(f"E-INTERP-4 {cid} failed: {e}")

    write_csv(e1_rows, os.path.join(OUT_DIR, "head_ablation_T1.csv"))
    write_csv(e2_rows, os.path.join(OUT_DIR, "residual_probing_T1.csv"))
    write_csv(e3_rows, os.path.join(OUT_DIR, "attention_patterns_T1.csv"))
    with open(os.path.join(OUT_DIR, "attention_patterns_T1_meta.json"), "w") as f:
        json.dump(e3_meta, f, indent=2)
    write_csv(e4_rows, os.path.join(OUT_DIR, "compression_T1.csv"))
    log("=== interp_t1.py DONE ===")
    logf.close()


if __name__ == "__main__":
    main()
