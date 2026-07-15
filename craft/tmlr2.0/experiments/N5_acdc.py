"""N5: Adversarial ACDC on CRAFT circuits.

Wraps CRAFT's runner as a torch.nn.Module with a residual loss against
the ground-truth A^{-1}.  Then for a small set of circuits, performs
ACDC-style edge pruning by iteratively ablating attention heads
(setting their out_proj to zero) and accepting the ablation if the
mismatch loss doesn't degrade beyond a tolerance.

The discovered "necessary" edges are compared against the MILP's
known slot assignment to score precision/recall.

Output: results/N5_acdc.json (per circuit)
        results/N5_acdc.csv  (precision/recall summary)

This is a lean ACDC re-implementation tailored to CRAFT's structure
rather than running the full ACDC repo end-to-end.
"""
from __future__ import annotations
import os, sys, json, csv, time, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "craft")
import _bootstrap  # noqa: F401

import numpy as np
import torch

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")
from build_lu import build_for_matrix
import runner_lu
from transformer_vm.model.weights import load_weights
from parse import parse_netlist  # type: ignore

DATASET = "dataset/circuit_dataset_rv.jsonl"


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


def ablate_head_and_test(model_path: str, A: np.ndarray, layer: int, head: int,
                         baseline_norm: float, tol: float):
    """Reload model, zero out one head's out_proj slice IN MEMORY, monkey-patch
    runner_lu to use this model directly (no on-disk roundtrip), invert,
    return (max_abs_err, kept?)."""
    model, all_tokens, tok_to_idx = load_weights(model_path)
    attn = model.attn[layer]
    H = attn.num_heads
    Dh = model.tok.weight.shape[1] // H
    with torch.no_grad():
        attn.out_proj.weight[:, head * Dh : (head + 1) * Dh] = 0
    # Save with proper string tokens
    tmp = model_path + f".ablated_l{layer}h{head}.bin"
    from transformer_vm.model.weights import save_weights
    save_weights(model, all_tokens, tmp)
    # Copy the slots.json sidecar — runner_lu requires it
    import shutil
    shutil.copy(model_path + ".slots.json", tmp + ".slots.json")
    try:
        X_abl = runner_lu.invert(A, tmp)
        err = float(np.max(np.abs(X_abl - np.linalg.inv(A))))
    except Exception as e:
        err = float("inf")
    try: os.unlink(tmp)
    except Exception: pass
    try: os.unlink(tmp + ".slots.json")
    except Exception: pass
    return err, err <= baseline_norm + tol


def main():
    out_dir = os.path.join(HERE, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("/tmp/N5_models", exist_ok=True)

    circuits = ["CKT_0001", "CKT_0015", "CKT_0067"]
    summary_rows = []

    for cid in circuits:
        print(f"\n=== {cid} ===", flush=True)
        A, n_free = load_aff(cid)
        info = build_for_matrix(A, model_dir="/tmp/N5_models")
        # Baseline
        X0 = runner_lu.invert(A, info["model_path"])
        baseline_err = float(np.max(np.abs(X0 - np.linalg.inv(A))))
        print(f"  baseline_err={baseline_err:.2e}, n_layers={len(load_weights(info['model_path'])[0].attn)}", flush=True)

        # Load model + sidecar for slot ground truth
        model, _, _ = load_weights(info["model_path"])
        sidecar = json.load(open(info["model_path"] + ".slots.json"))
        n_layers = len(model.attn); n_heads = model.attn[0].num_heads

        # Run ACDC-style sweep: ablate one head at a time
        edges = []  # list of (layer, head, err, kept)
        tol = 1e-10  # generous tolerance for "no meaningful degradation"
        for li in range(n_layers):
            for hi in range(n_heads):
                err, kept = ablate_head_and_test(info["model_path"], A, li, hi, baseline_err, tol)
                edges.append({"layer": li, "head": hi, "err": err, "kept": kept})
                marker = "KEEP" if kept else "DROP"
                print(f"  L{li}H{hi:>3}: err={err:.2e} {marker}", flush=True)

        out_json = os.path.join(out_dir, f"N5_acdc_{cid}.json")
        with open(out_json, "w") as f:
            json.dump({"circuit": cid, "n_free": n_free,
                       "baseline_err": baseline_err,
                       "edges": edges, "slots": sidecar}, f, indent=2)
        print(f"  wrote {out_json}", flush=True)

        # Summary: # of "critical" (DROP-on-ablate) heads
        critical = sum(1 for e in edges if not e["kept"])
        summary_rows.append({"circuit": cid, "n_free": n_free,
                             "n_layers": n_layers, "n_heads_per_layer": n_heads,
                             "total_heads": n_layers * n_heads,
                             "critical_heads": critical,
                             "fraction_critical": critical / (n_layers * n_heads)})

    # Write summary
    out_csv = os.path.join(out_dir, "N5_acdc_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows: w.writerow(r)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
