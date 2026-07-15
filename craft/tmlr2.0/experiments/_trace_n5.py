"""Capture a per-layer activation trace for a 5x5 LU inversion.

Solves Ax = e_0 (the first column of A^-1) and prints, for every token in
the sequence, the residual-stream slot values that matter to the algorithm.
Writes a JSON-ish text file the explainer Markdown can quote.
"""
from __future__ import annotations

import json
import os
import sys
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401, E402

from build_lu import build_for_matrix  # noqa: E402
from runner_lu import _build_x, CACHE_CLASS, _vcache_patch_lu  # noqa: E402
from transformer_vm.model.weights import load_weights  # noqa: E402
from lu_factor import doolittle  # noqa: E402


def main(out_path: str = "trace_n5.json"):
    np.random.seed(42)
    n = 5
    A = 5.0 * np.eye(n) + 0.3 * np.random.randn(n, n)
    L, U = doolittle(A)

    os.makedirs("/tmp/trace", exist_ok=True)
    r = build_for_matrix(A, model_dir="/tmp/trace")
    sidecar = json.load(open(r["model_path"] + ".slots.json"))
    slot_x_value = sidecar["slot_x_value_slot"]
    slot_b_value = sidecar["slot_b_value_slot"]
    slot_y_input = sidecar["slot_y_input_slot"]
    slot_x_new = sidecar["slot_x_new"]
    named = sidecar["slot_of_named"]

    model, all_tokens, tok_to_idx = load_weights(r["model_path"])
    model.eval()
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = CACHE_CLASS(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    b = np.zeros(n); b[0] = 1.0   # e_0
    y_buf = [0.0] * n
    x_result = [0.0] * n

    tokens = (
        ["start"]
        + [f"init_{i}" for i in range(n)]
        + [f"fwd_{i}"  for i in range(n)]
        + [f"bck_{i}"  for i in range(n - 1, -1, -1)]
        + ["halt"]
    )

    trace = {
        "matrix_A": A.tolist(),
        "L": L.tolist(),
        "U": U.tolist(),
        "b": b.tolist(),
        "x_ref": np.linalg.solve(A, b).tolist(),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": model.tok.weight.shape[1],
        "d_ffn": model.ff_in[0].weight.shape[0] // 2,
        "vocab": list(all_tokens),
        "slots": {
            "x_value_slot": slot_x_value,
            "b_value_slot": slot_b_value,
            "y_input_slot": slot_y_input,
            "x_new":        slot_x_new,
        },
        "slot_of_named": named,
        "tokens": [],
    }

    pos = 0
    with torch.no_grad():
        for tok in tokens:
            tok_idx = tok_to_idx[tok]
            x_init = _build_x(model, tok_idx, pos)

            entry = {"pos": pos, "tok": tok}
            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_b_value] = x_init[slot_b_value] + float(b[i])
                entry["inject_slot_b_value"] = float(b[i])
            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_y_input] = x_init[slot_y_input] + float(y_buf[i])
                entry["inject_slot_y_input"] = float(y_buf[i])

            entry["x_init_at_slots"] = {
                "slot_x_value": float(x_init[slot_x_value].item()),
                "slot_b_value": float(x_init[slot_b_value].item()),
                "slot_y_input": float(x_init[slot_y_input].item()),
                "slot_x_new":   float(x_init[slot_x_new].item()),
            }

            x = x_init.clone()
            entry["per_layer"] = []
            for li, (attn, ff_in, ff_out) in enumerate(
                zip(model.attn, model.ff_in, model.ff_out, strict=True)
            ):
                pre_attn = float(x[slot_x_new].item())
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                cache_out = cache.layer_step(li, k, q, v)
                attn_delta = attn.out_proj(cache_out)
                x = x + attn_delta
                post_attn = float(x[slot_x_new].item())

                gate, val = ff_in(x).chunk(2, dim=-1)
                ffn_delta = ff_out(F.relu(gate) * val)
                x = x + ffn_delta
                post_ffn = float(x[slot_x_new].item())
                entry["per_layer"].append({
                    "layer": li,
                    "pre_attn_slot_x_new": pre_attn,
                    "attn_delta_slot_x_new":  float(attn_delta[slot_x_new].item()),
                    "post_attn_slot_x_new":   post_attn,
                    "ffn_delta_slot_x_new":   float(ffn_delta[slot_x_new].item()),
                    "post_ffn_slot_x_new":    post_ffn,
                })

            final_x_new = float(x[slot_x_new].item())
            entry["final_slot_x_new"] = final_x_new

            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                y_buf[i] = final_x_new
                entry["saved_to"] = f"y_buf[{i}]"
                entry["saved_value"] = y_buf[i]
                _vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value,
                    slot_x_new=slot_x_new,
                    new_value=y_buf[i],
                    extra={slot_b_value: float(b[i])},
                )
                entry["v_cache_patch"] = {
                    "slot_x_value": float(y_buf[i]),
                    "slot_b_value": float(b[i]),
                    "slot_x_new":   float(y_buf[i]),
                }
            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_result[i] = final_x_new
                entry["saved_to"] = f"x_result[{i}]"
                entry["saved_value"] = x_result[i]
                _vcache_patch_lu(
                    model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value,
                    slot_x_new=slot_x_new,
                    new_value=x_result[i],
                    extra={slot_y_input: float(y_buf[i])},
                )
                entry["v_cache_patch"] = {
                    "slot_x_value": float(x_result[i]),
                    "slot_y_input": float(y_buf[i]),
                    "slot_x_new":   float(x_result[i]),
                }

            trace["tokens"].append(entry)
            pos += 1

    trace["y_buf"] = y_buf
    trace["x_result"] = x_result
    trace["max_abs_err_vs_numpy"] = float(np.max(np.abs(np.array(x_result) - np.array(trace["x_ref"]))))

    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    print("wrote", out_path)
    print("y_buf  =", y_buf)
    print("x      =", x_result)
    print("x_ref  =", trace["x_ref"])
    print("max_err=", trace["max_abs_err_vs_numpy"])


if __name__ == "__main__":
    main()
