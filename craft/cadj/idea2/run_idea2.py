"""Run a single circuit using the idea2 readout model.

Inference is identical to CRAFT's direct algorithm — the model is a standard
direct readout transformer, just with S baked in from the LU transformer rather
than numpy.linalg.solve. Token sequence: start, (skip, v_init_j)*N, readout,
<PRED>, halt  — 2N+4 tokens.

Usage:
    python run_idea2.py CKT_0001
    python run_idea2.py CKT_0001 --verbose
"""
from __future__ import annotations

import os
import sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
MODEL_DIR = _CLAUDE_FILES


def run_circuit_idea2(
    cid: str,
    tol: float = 0.05,
    v_step: int | None = None,
    k_levels: int | None = None,
    verbose: bool = False,
    model_dir: str = MODEL_DIR,
) -> tuple[str, float, float, float]:
    """Run idea2 readout model for cid. Returns (status, pred_v, truth_v, elapsed)."""
    import runner
    model_path = os.path.join(model_dir, f"model_{cid}_idea2.bin")
    return runner.run_circuit(
        cid,
        model_dir=model_dir,
        T=None,
        tol=tol,
        verbose=verbose,
        algorithm="direct",
        v_step=v_step,
        k_levels=k_levels,
        model_path_override=model_path,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run idea2 circuit inference")
    ap.add_argument("circuit_id", help="e.g. CKT_0001")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    status, pred_v, truth, elapsed = run_circuit_idea2(
        args.circuit_id, tol=args.tol, verbose=args.verbose, model_dir=args.model_dir,
    )
    err = abs(pred_v - truth)
    print(f"{status}  {args.circuit_id}  pred={pred_v:.4f}  truth={truth:.4f}  "
          f"err={err:.4f}  time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
