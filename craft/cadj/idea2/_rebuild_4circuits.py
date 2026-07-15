"""Rebuild and re-run the 4 circuits that failed, then re-run the full sweep."""
import os, sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

from _server_build_idea2 import build_for_circuit
from _server_run_idea2 import run_circuit_idea2

CIRCUITS = ["CKT_0039", "CKT_0067", "CKT_0068", "CKT_0130"]

for cid in CIRCUITS:
    print(f"\n=== {cid} ===", flush=True)
    info = build_for_circuit(cid, force=True)
    print("Build:", {k: v for k, v in info.items() if k != "model_path"}, flush=True)
    status, pred_v, truth, infer_s, n_tokens = run_circuit_idea2(cid)
    err = abs(pred_v - truth)
    print(f"{status}  pred={pred_v:.4f}  truth={truth:.4f}  err={err:.4f}  infer={infer_s:.1f}s  tokens={n_tokens}", flush=True)
