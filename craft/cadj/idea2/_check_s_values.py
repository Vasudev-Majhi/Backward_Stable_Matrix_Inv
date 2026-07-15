"""Diagnose the 4 failing circuits by comparing S from numpy vs LU transformer."""
import sys, os, json
import numpy as np

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401

from parse import parse_netlist
from direct_reference import build_sensitivity_matrix
from _server_build_idea2 import _conductance_matrix, invert_via_lu_transformer, _lu_model_path, LU_MODEL_DIR

DATASET = os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl")

def load(cid):
    with open(DATASET) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                return c

COEF_SCALE = 10000

for cid in ["CKT_0039", "CKT_0067", "CKT_0068", "CKT_0130"]:
    c = load(cid)
    pc = parse_netlist(c["Netlist"])
    target = int(c["Target_Node"])
    truth = float(c["Ground_Truth_Vout"])
    N = pc.num_nodes

    free  = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]
    fi, pi = np.array(free), np.array(fixed)

    A    = _conductance_matrix(pc)
    A_FF = A[np.ix_(fi, fi)]
    A_FP = A[np.ix_(fi, pi)]

    # numpy S
    _, _, S_numpy = build_sensitivity_matrix(pc)

    # LU transformer S
    lu_path = _lu_model_path(A_FF, os.path.join(LU_MODEL_DIR, cid))
    A_FF_inv_lu, _ = invert_via_lu_transformer(A_FF, cid, lu_path)
    S_lu = -A_FF_inv_lu @ A_FP

    if target not in free:
        print(f"{cid}: target fixed, skip"); continue
    ti = free.index(target)

    v_P = [pc.fixed_voltage[j] for j in fixed]
    v_exact_numpy = float(S_numpy[ti] @ v_P)
    v_exact_lu    = float(S_lu[ti]    @ v_P)

    # What COEF_SCALE rounding produces
    s_int_numpy = [int(round(max(S_numpy[ti, k], 0.0) * COEF_SCALE)) for k in range(len(fixed))]
    s_int_lu    = [int(round(max(S_lu[ti,    k], 0.0) * COEF_SCALE)) for k in range(len(fixed))]

    print(f"\n{cid}  truth={truth:.4f}  n_free={len(free)}")
    print(f"  v_exact_numpy={v_exact_numpy:.6f}  v_exact_lu={v_exact_lu:.6f}")
    print(f"  S_numpy[target,:]  = {[round(float(x),6) for x in S_numpy[ti]]}")
    print(f"  S_lu[target,:]     = {[round(float(x),6) for x in S_lu[ti]]}")
    print(f"  s_int_numpy        = {s_int_numpy}")
    print(f"  s_int_lu           = {s_int_lu}")
    print(f"  max_S_err (LU-numpy) = {float(np.max(np.abs(S_lu - S_numpy))):.2e}")
