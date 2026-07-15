"""N5 v2: In-memory ACDC head ablation on CRAFT models.

Replaces the v1 save/reload roundtrip (which silently corrupts state_dicts
for small models, invalidating CKT_0001 and CKT_0015) with an in-memory
ablation loop:

  1. Load the compiled model once.
  2. Clone the original out_proj weight matrix per layer (small, OK in memory).
  3. For each (layer, head): zero the relevant out_proj slice in-place,
     invert column-by-column via solve_column_lu, restore the slice.

Output:
    results/N5_v2/N5_acdc_<cid>.json   (per circuit edge list)
    results/N5_v2/N5_acdc_summary.csv  (critical-head counts per circuit)

Usage (server, in the craft_release venv):
    cd ~/craft_release/craft/tmlr2.0/experiments
    ~/miniforge3/bin/python3 N5_acdc_v2.py
"""
from __future__ import annotations
import os, sys, json, csv, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "matrix_inversion", "inversion2"))

import _bootstrap  # noqa: F401
import numpy as np
import torch

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")
from build_lu import build_for_matrix
import runner_lu
from transformer_vm.model.weights import load_weights
from parse import parse_netlist  # type: ignore

DATASET = os.path.join(HOME, "craft_release", "dataset", "circuit_dataset_rv.jsonl")


def load_aff(cid: str):
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
                return A[np.ix_(free, free)], len(free)
    raise KeyError(cid)


def _invert_with_loaded_model(A, model, tok_to_idx, sidecar):
    """Same logic as runner_lu.invert but takes a pre-loaded model object,
    so we can ablate in-memory without filesystem round-trips."""
    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])
    n_side = int(sidecar["n"])
    n = A.shape[0]
    if n != n_side:
        raise ValueError(f"Matrix n={n} != sidecar n={n_side}")
    model.eval()
    X = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        e_j = np.zeros(n, dtype=np.float64); e_j[j] = 1.0
        X[:, j] = runner_lu.solve_column_lu(
            model, tok_to_idx, n,
            slot_x_value, slot_b_value, slot_y_input, slot_x_new,
            b=e_j, verbose=False,
        )
    return X


def run_one_circuit(cid, out_dir, tol=1e-10):
    print(f"\n=== {cid} ===", flush=True)
    A, n_free = load_aff(cid)
    info = build_for_matrix(A, model_dir="/tmp/N5_v2_models")
    sidecar = json.load(open(info["model_path"] + ".slots.json"))

    # Load ONCE, ablate in place, restore.
    model, all_tokens, tok_to_idx = load_weights(info["model_path"])
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    Dh_per_head = model.tok.weight.shape[1] // n_heads

    # Baseline
    X0 = _invert_with_loaded_model(A, model, tok_to_idx, sidecar)
    baseline_err = float(np.max(np.abs(X0 - np.linalg.inv(A))))
    print(f"  baseline_err={baseline_err:.2e}  n_layers={n_layers}  n_heads={n_heads}  Dh={Dh_per_head}", flush=True)

    # Snapshot of original weights (one clone per layer).
    original = [model.attn[li].out_proj.weight.detach().clone() for li in range(n_layers)]

    edges = []
    for li in range(n_layers):
        attn = model.attn[li]
        H = attn.num_heads
        Dh = model.tok.weight.shape[1] // H
        for hi in range(H):
            # Ablate in-place
            with torch.no_grad():
                attn.out_proj.weight[:, hi * Dh:(hi + 1) * Dh] = 0
            try:
                X_abl = _invert_with_loaded_model(A, model, tok_to_idx, sidecar)
                err = float(np.max(np.abs(X_abl - np.linalg.inv(A))))
            except Exception as e:
                err = float("inf")
                print(f"  L{li}H{hi:>3}: EXCEPTION {e!r}", flush=True)
            kept = err <= baseline_err + tol
            edges.append({"layer": li, "head": hi, "err": err, "kept": kept})
            marker = "KEEP" if kept else "DROP"
            print(f"  L{li}H{hi:>3}: err={err:.2e} {marker}", flush=True)
            # Restore
            with torch.no_grad():
                attn.out_proj.weight.copy_(original[li])

    out_json = os.path.join(out_dir, f"N5_acdc_{cid}.json")
    with open(out_json, "w") as f:
        json.dump({"circuit": cid, "n_free": n_free,
                   "baseline_err": baseline_err,
                   "n_layers": n_layers, "n_heads": n_heads,
                   "edges": edges, "slots": sidecar}, f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    critical = sum(1 for e in edges if not e["kept"])
    return {
        "circuit": cid, "n_free": n_free,
        "baseline_err": baseline_err,
        "n_layers": n_layers, "n_heads_per_layer": n_heads,
        "total_heads": n_layers * n_heads,
        "critical_heads": critical,
        "fraction_critical": critical / (n_layers * n_heads),
        "per_layer_critical": [
            sum(1 for e in edges if e["layer"] == li and not e["kept"])
            for li in range(n_layers)
        ],
    }


def main():
    out_dir = os.path.join(HERE, "..", "results", "N5_v2")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("/tmp/N5_v2_models", exist_ok=True)

    # Add CKT_0068 and CKT_0130 to widen statistical claim beyond n=1 circuit.
    circuits = ["CKT_0001", "CKT_0015", "CKT_0067", "CKT_0068"]
    # CKT_0130 has n_free=38 (105*38*3 ablations, too slow for now); add later if time.

    summary_rows = []
    for cid in circuits:
        try:
            row = run_one_circuit(cid, out_dir)
            summary_rows.append(row)
        except Exception as e:
            print(f"  FAILED on {cid}: {e!r}", flush=True)

    out_csv = os.path.join(out_dir, "N5_acdc_summary.csv")
    if summary_rows:
        with open(out_csv, "w", newline="") as f:
            keys = [k for k in summary_rows[0].keys() if k != "per_layer_critical"]
            keys += ["per_layer_critical"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                r2 = dict(r); r2["per_layer_critical"] = json.dumps(r["per_layer_critical"])
                w.writerow(r2)
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
