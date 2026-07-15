"""N1: Symbolic execution trace for real paper circuits CKT_0001 and CKT_0067.

Adapts inversion2/_trace_n5.py: instead of a random 5x5 matrix, loads the
actual A_FF from the paper circuit dataset and traces the LU inversion
token-by-token, slot-by-slot, per-layer.

Output: results/N1_trace_<CKT_ID>.json
"""
from __future__ import annotations
import json, os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")

from build_lu import build_for_matrix
from runner_lu import _build_x, CACHE_CLASS, _vcache_patch_lu
from transformer_vm.model.weights import load_weights
from lu_factor import doolittle

# Reuse parse.py from craft for circuit loading
sys.path.insert(0, "craft")
from parse import parse_netlist  # type: ignore

DATASET = "dataset/circuit_dataset_rv.jsonl"


def load_circuit_aff(cid: str):
    """Return (A_FF, n_free) for circuit cid."""
    with open(DATASET) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                pc = parse_netlist(c["Netlist"])
                N = pc.num_nodes
                A = np.zeros((N, N), dtype=np.float64)
                for (a, b, ohms) in pc.resistors:
                    g = 1.0 / ohms
                    A[a, a] += g; A[b, b] += g
                    A[a, b] -= g; A[b, a] -= g
                free = [i for i in range(N) if not pc.is_fixed[i]]
                A_FF = A[np.ix_(free, free)]
                return A_FF, len(free)
    raise KeyError(cid)


def trace_one(cid: str, out_path: str):
    A, n = load_circuit_aff(cid)
    L, U = doolittle(A)
    r = build_for_matrix(A, model_dir="/tmp/N1_models")
    sidecar = json.load(open(r["model_path"] + ".slots.json"))
    slot_x_value = sidecar["slot_x_value_slot"]
    slot_b_value = sidecar["slot_b_value_slot"]
    slot_y_input = sidecar["slot_y_input_slot"]
    slot_x_new   = sidecar["slot_x_new"]

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

    # Trace only column j=0 (first column of A^-1) for compactness
    b = np.zeros(n); b[0] = 1.0
    y_buf = [0.0] * n
    x_result = [0.0] * n

    tokens = (["start"] + [f"init_{i}" for i in range(n)]
              + [f"fwd_{i}" for i in range(n)]
              + [f"bck_{i}" for i in range(n - 1, -1, -1)]
              + ["halt"])

    trace = {
        "circuit_id": cid, "n_free": n, "column": 0,
        "matrix_A": A.tolist(), "L": L.tolist(), "U": U.tolist(),
        "b": b.tolist(), "x_ref": np.linalg.solve(A, b).tolist(),
        "n_layers": n_layers, "n_heads": n_heads,
        "d_model": model.tok.weight.shape[1],
        "slots": {"x_value": slot_x_value, "b_value": slot_b_value,
                  "y_input": slot_y_input, "x_new": slot_x_new},
        "tokens": [],
    }

    pos = 0
    with torch.no_grad():
        for tok in tokens:
            x_init = _build_x(model, tok_to_idx[tok], pos)
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
                zip(model.attn, model.ff_in, model.ff_out, strict=True)):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                cache_out = cache.layer_step(li, k, q, v)
                attn_delta = attn.out_proj(cache_out)
                x = x + attn_delta
                gate, val = ff_in(x).chunk(2, dim=-1)
                ffn_delta = ff_out(F.relu(gate) * val)
                x = x + ffn_delta
                entry["per_layer"].append({
                    "layer": li,
                    "attn_delta_x_new": float(attn_delta[slot_x_new].item()),
                    "ffn_delta_x_new": float(ffn_delta[slot_x_new].item()),
                    "post_ffn_x_new": float(x[slot_x_new].item()),
                })

            final_x_new = float(x[slot_x_new].item())
            entry["final_slot_x_new"] = final_x_new

            if tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                y_buf[i] = final_x_new
                _vcache_patch_lu(model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value, slot_x_new=slot_x_new,
                    new_value=y_buf[i], extra={slot_b_value: float(b[i])})
            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_result[i] = final_x_new
                _vcache_patch_lu(model, cache, x_init, pos, n_layers,
                    slot_x_value=slot_x_value, slot_x_new=slot_x_new,
                    new_value=x_result[i], extra={slot_y_input: float(y_buf[i])})

            trace["tokens"].append(entry)
            pos += 1

    trace["y_buf"] = y_buf
    trace["x_result"] = x_result
    trace["max_abs_err_vs_numpy"] = float(
        np.max(np.abs(np.array(x_result) - np.array(trace["x_ref"]))))
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"wrote {out_path}  max_err_vs_numpy={trace['max_abs_err_vs_numpy']:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", nargs="+", default=["CKT_0001", "CKT_0067"])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs("/tmp/N1_models", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    for cid in args.circuits:
        out_path = os.path.join(args.out_dir, f"N1_trace_{cid}.json")
        try:
            trace_one(cid, out_path)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {cid}: {e}")


if __name__ == "__main__":
    main()
