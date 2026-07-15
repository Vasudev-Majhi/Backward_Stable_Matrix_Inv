"""T1.9 — HARD_K sensitivity + dtype downcast sweep on CRAFT models.

For a stratified subset of 6 circuits we sweep:
  - HARD_K ∈ {1e4, 1e6, 1e8, 1e10, 1e12}
  - dtype ∈ {float64, float32, bfloat16}

Method:
  1. Monkey-patch transformer_vm.model.weights.HARD_K before build_for_matrix.
  2. After build + load, optionally downcast model weights to float32/bf16.
  3. Run runner_lu.invert; record max_abs_err vs np.linalg.inv(A).

Establishes the algebraic boundary of CRAFT's softmax-saturation contract.

Output: results/T9/hardk_sweep.csv
"""
from __future__ import annotations
import os, sys, csv, json, time, math

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "matrix_inversion", "inversion2"))

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")
import numpy as np
import torch
import _bootstrap  # noqa: F401

from build_lu import build_for_matrix
import runner_lu
from transformer_vm.model import weights as wmod
from transformer_vm.model.weights import load_weights
from parse import parse_netlist  # type: ignore

DATASET = os.path.join(HOME, "craft_release", "dataset", "circuit_dataset_rv.jsonl")

# Stratified subset across n_free values.
SUBSET = ["CKT_0001", "CKT_0015", "CKT_0067", "CKT_0068", "CKT_0091", "CKT_0130"]
HARD_K_GRID = [1e4, 1e6, 1e8, 1e10, 1e12]
DTYPE_GRID = ["float64", "float32", "bfloat16"]
DTYPE_OF = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}


def load_aff(cid):
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


def cast_model(model, torch_dtype):
    """Cast all model parameters to a given torch dtype."""
    for p in model.parameters():
        p.data = p.data.to(torch_dtype)


def run_one(cid, hard_k, dtype_name, work_dir):
    """Build with given HARD_K, optionally downcast, invert, return error."""
    A, n_free = load_aff(cid)
    # Monkey-patch HARD_K and rebuild.
    wmod.HARD_K = hard_k
    model_dir = os.path.join(work_dir, f"hk{hard_k:.0e}_{dtype_name}")
    os.makedirs(model_dir, exist_ok=True)
    t0 = time.time()
    info = build_for_matrix(A, model_dir=model_dir)
    build_s = time.time() - t0
    # Load and optionally downcast.
    model, all_tokens, tok_to_idx = load_weights(info["model_path"])
    if dtype_name != "float64":
        cast_model(model, DTYPE_OF[dtype_name])
    # Patch runner_lu's load_weights to return our (already-cast) model.
    # Easier: replicate solve_column_lu manually using runner_lu.solve_column_lu.
    sidecar = json.load(open(info["model_path"] + ".slots.json"))
    slot_x_value = int(sidecar["slot_x_value_slot"])
    slot_b_value = int(sidecar["slot_b_value_slot"])
    slot_y_input = int(sidecar["slot_y_input_slot"])
    slot_x_new   = int(sidecar["slot_x_new"])
    n_side = int(sidecar["n"])
    assert A.shape[0] == n_side
    model.eval()
    t0 = time.time()
    X = np.zeros((n_side, n_side), dtype=np.float64)
    try:
        for j in range(n_side):
            e_j = np.zeros(n_side, dtype=np.float64); e_j[j] = 1.0
            X[:, j] = runner_lu.solve_column_lu(
                model, tok_to_idx, n_side,
                slot_x_value, slot_b_value, slot_y_input, slot_x_new,
                b=e_j, verbose=False,
            )
        infer_s = time.time() - t0
        max_err = float(np.max(np.abs(X - np.linalg.inv(A))))
    except Exception as e:
        infer_s = float("nan")
        max_err = float("inf")
        print(f"    EXC: {e!r}", flush=True)
    return {
        "circuit": cid, "n_free": n_free,
        "hard_k": hard_k, "dtype": dtype_name,
        "build_s": build_s, "infer_s": infer_s,
        "max_err": max_err,
    }


def main():
    out_dir = os.path.join(HERE, "..", "results", "T9")
    os.makedirs(out_dir, exist_ok=True)
    work_dir = "/tmp/T9_models"
    os.makedirs(work_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "hardk_sweep.csv")
    rows = []
    fields = ["circuit", "n_free", "hard_k", "dtype", "build_s", "infer_s", "max_err"]
    for cid in SUBSET:
        for hk in HARD_K_GRID:
            for dt in DTYPE_GRID:
                print(f"\n[{cid}] HARD_K={hk:.0e}  dtype={dt}", flush=True)
                try:
                    r = run_one(cid, hk, dt, work_dir)
                except Exception as e:
                    r = {"circuit": cid, "n_free": "?", "hard_k": hk, "dtype": dt,
                         "build_s": "", "infer_s": "", "max_err": "inf"}
                    print(f"    BUILD/RUN FAILED: {e!r}", flush=True)
                rows.append(r)
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                    for x in rows: w.writerow(x)
                print(f"    max_err={r['max_err']}", flush=True)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
