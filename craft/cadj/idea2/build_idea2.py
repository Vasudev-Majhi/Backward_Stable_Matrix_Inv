"""Build per-circuit readout model with S from the LU inversion transformer.

Pipeline for each circuit CKT_XXXX:
  1. Extract A_FF, A_FP from the conductance matrix (numpy only, no linalg).
  2. Build the LU inversion transformer for A_FF via subprocess → inversion2.
  3. Run the LU transformer to get A_FF⁻¹ via subprocess → inversion2.
  4. Compute S = -A_FF⁻¹ @ A_FP  (pure matrix multiply, no linalg.solve/inv).
  5. Bake S into the CRAFT readout model via DirectCircuitMachine(S_override=S).

Paper claim: no numpy.linalg.solve or numpy.linalg.inv is called in this file.
The matrix inversion runs entirely inside the transformer's forward passes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Path setup (CRAFT environment) ────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))   # cadj/idea2/
_CADJ         = os.path.dirname(_HERE)                        # cadj/
_CLAUDE_FILES = os.path.dirname(_CADJ)                        # craft/

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401 — adds CRAFT's transformer-vm to sys.path

# ── Constants ─────────────────────────────────────────────────────────────────
INVERSION2_DIR = os.path.normpath(
    os.path.join(os.path.dirname(_CLAUDE_FILES),
                 "..", "Matrix_inversion", "inversion2")
)
# Fallback to absolute path if the relative path doesn't exist.
if not os.path.isdir(INVERSION2_DIR):
    INVERSION2_DIR = r"matrix_inversion/inversion2"

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
MODEL_DIR    = _CLAUDE_FILES                              # readout models
LU_MODEL_DIR = os.path.join(_HERE, "lu_models")          # LU weights per-circuit


def load_circuit(cid: str) -> dict:
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                return c
    raise KeyError(f"Circuit {cid} not found in {DATASET_PATH}")


def _conductance_matrix(pc) -> np.ndarray:
    N = pc.num_nodes
    A = np.zeros((N, N), dtype=np.float64)
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        A[a, a] += g; A[b, b] += g; A[a, b] -= g; A[b, a] -= g
    return A


def _lu_model_path(A_FF: np.ndarray, cid_lu_dir: str) -> str:
    h = hashlib.sha1(A_FF.tobytes()).hexdigest()[:12]
    return os.path.join(cid_lu_dir, f"model_{h}_lu.bin")


def build_lu_transformer(A_FF: np.ndarray, cid: str) -> str:
    """Build LU transformer for A_FF via subprocess. Returns model_path."""
    cid_lu_dir = os.path.join(LU_MODEL_DIR, cid)
    os.makedirs(cid_lu_dir, exist_ok=True)
    model_path = _lu_model_path(A_FF, cid_lu_dir)

    if os.path.exists(model_path) and os.path.exists(model_path + ".slots.json"):
        log.info("LU model cached: %s", model_path)
        return model_path

    build_script = os.path.join(INVERSION2_DIR, "build_lu.py")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        A_FF_path = tf.name
    try:
        np.save(A_FF_path, A_FF)
        log.info("Building LU transformer for %s (n=%d)...", cid, A_FF.shape[0])
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, build_script,
             "--matrix", A_FF_path,
             "--model-dir", cid_lu_dir],
            cwd=INVERSION2_DIR,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"build_lu.py failed for {cid} (exit {r.returncode}):\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        log.info("LU build done in %.1fs", time.time() - t0)
    finally:
        os.unlink(A_FF_path)

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Expected model not found after build: {model_path}\n"
            f"build_lu.py stdout:\n{r.stdout}"
        )
    return model_path


def invert_via_lu_transformer(A_FF: np.ndarray, cid: str, lu_model_path: str) -> np.ndarray:
    """Run LU transformer to compute A_FF⁻¹. Returns float64 array.

    All arithmetic happens inside the transformer's forward passes.
    No numpy.linalg is called in this function.
    """
    runner_script = os.path.join(INVERSION2_DIR, "runner_lu.py")
    cid_lu_dir = os.path.dirname(lu_model_path)

    with tempfile.TemporaryDirectory() as tmp:
        A_FF_path     = os.path.join(tmp, "A_FF.npy")
        A_FF_inv_path = os.path.join(tmp, "A_FF_inv.npy")
        np.save(A_FF_path, A_FF)

        log.info("Running LU transformer for %s (n=%d)...", cid, A_FF.shape[0])
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, runner_script,
             "--matrix", A_FF_path,
             "--model-dir", cid_lu_dir,
             "--out", A_FF_inv_path],
            cwd=INVERSION2_DIR,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"runner_lu.py failed for {cid} (exit {r.returncode}):\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        log.info("LU run done in %.1fs", time.time() - t0)

        A_FF_inv = np.load(A_FF_inv_path)

    return A_FF_inv


def build_for_circuit(
    cid: str,
    out_path: str | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
    force: bool = False,
) -> dict:
    """Build idea2 readout model for circuit cid.

    Returns dict with model_path and build stats.
    No numpy.linalg.solve or numpy.linalg.inv is called here.
    """
    from direct_interpreter import DirectCircuitMachine
    from parse import parse_netlist
    from transformer_vm.model.weights import build_model, save_weights

    if out_path is None:
        out_path = os.path.join(MODEL_DIR, f"model_{cid}_idea2.bin")

    if os.path.exists(out_path) and not force:
        log.info("Readout model cached, skipping: %s", out_path)
        return {"model_path": out_path, "cached": True}

    c = load_circuit(cid)
    netlist = c["Netlist"]
    target  = int(c["Target_Node"])

    pc = parse_netlist(netlist)
    N  = pc.num_nodes

    # Step 1-2: conductance matrix + partition
    A    = _conductance_matrix(pc)
    free  = [i for i in range(N) if not pc.is_fixed[i]]
    fixed = [i for i in range(N) if pc.is_fixed[i]]

    if not free or not fixed:
        raise ValueError(f"{cid}: degenerate circuit (free={len(free)}, fixed={len(fixed)})")

    fi = np.array(free)
    pi = np.array(fixed)
    A_FF = A[np.ix_(fi, fi)]   # n_free × n_free  ← what we invert
    A_FP = A[np.ix_(fi, pi)]   # n_free × n_fixed

    # Step 3: build LU transformer for A_FF (subprocess → inversion2)
    lu_model_path = build_lu_transformer(A_FF, cid)

    # Step 4: run LU transformer → A_FF⁻¹  (subprocess → inversion2)
    A_FF_inv = invert_via_lu_transformer(A_FF, cid, lu_model_path)

    # Step 5: S = -A_FF⁻¹ @ A_FP  (pure matmul, no linalg)
    S = -A_FF_inv @ A_FP   # (n_free × n_fixed)
    log.info("S computed: shape=%s  max|S|=%.6f", S.shape, float(np.max(np.abs(S))))

    # Step 6: bake S into CRAFT readout model
    machine = DirectCircuitMachine(
        pc, target, v_step=v_step, k_levels=k_levels, S_override=S,
    )
    pg, _ = machine.build()

    t1 = time.time()
    model, all_tokens, _, _ = build_model(program_graph=pg)
    log.info("Readout weights built in %.2fs", time.time() - t1)

    actual_d_model  = model.tok.weight.shape[1]
    actual_n_layers = len(model.attn)
    actual_d_ffn    = model.ff_in[0].weight.shape[0] // 2
    n_params = sum(p.numel() for p in model.parameters())

    save_weights(model, all_tokens, out_path)
    log.info("Saved: %s  (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    return {
        "circuit": cid,
        "N": N,
        "n_free": len(free),
        "n_fixed": len(fixed),
        "n_layers": actual_n_layers,
        "d_model": actual_d_model,
        "d_ffn": actual_d_ffn,
        "vocab": len(all_tokens),
        "n_params": n_params,
        "model_path": out_path,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build idea2 readout model (LU transformer inversion)")
    ap.add_argument("circuit_id", help="e.g. CKT_0001")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="Rebuild even if model exists")
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    args = ap.parse_args()

    result = build_for_circuit(
        args.circuit_id, out_path=args.out, force=args.force,
        v_step=args.v_step, k_levels=args.k_levels,
    )
    print(f"\nBuild complete: {result}")


if __name__ == "__main__":
    main()
