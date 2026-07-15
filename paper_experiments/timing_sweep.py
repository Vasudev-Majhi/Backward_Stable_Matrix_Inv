"""Full-dataset (154 circuit) build+inference timing sweep with NO cache.

For each circuit CKT_XXXX:
  1. Delete cached LU and readout models (if any)
  2. Call build_for_circuit(force=True) to rebuild both transformers
     - records lu_build_s (T1 build), lu_infer_s (T1 inference), readout_build_s (T2 build)
  3. Run readout inference, record readout_infer_s (T2 inference) and pred / pass_fail
  4. Append a row to CSV

Designed to be re-runnable (skips circuits already in the output CSV).
Run from inside ~/craft_release/craft/cadj/idea2/.

Outputs:
  ~/craft_release/craft/paper_experiments/results/timing_sweep.csv
  ~/craft_release/craft/paper_experiments/results/timing_sweep.log
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import sys
import time
import traceback

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

from _server_build_idea2 import (  # noqa: E402
    build_for_circuit, LU_MODEL_DIR, MODEL_DIR,
)
from _server_run_idea2 import run_circuit_idea2  # noqa: E402

DATASET = os.path.join(_REPO_ROOT, "dataset/circuit_dataset_rv.jsonl")
OUT_DIR = os.path.join(_REPO_ROOT, "craft/paper_experiments/results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "timing_sweep.csv")
LOG_PATH = os.path.join(OUT_DIR, "timing_sweep.log")

FIELDS = [
    "circuit_id", "complexity", "N", "n_free", "n_fixed",
    "T1_build_s", "T1_infer_s",
    "T2_build_s", "T2_infer_s",
    "T2_n_tokens",
    "T1_n_layers", "T1_d_model", "T1_n_params",
    "T2_n_layers", "T2_d_model", "T2_n_params",
    "pred_v", "truth_v", "abs_err_v",
    "pass_50mv", "pass_75mv",
    "status", "error",
]


def load_dataset():
    with open(DATASET) as f:
        return [json.loads(line) for line in f]


def already_done(out_csv: str) -> set[str]:
    if not os.path.exists(out_csv):
        return set()
    done = set()
    with open(out_csv) as f:
        for r in csv.DictReader(f):
            if r.get("status") in ("OK", "ERROR"):
                done.add(r["circuit_id"])
    return done


def clear_cache(cid: str):
    """Delete LU + readout model files so a fresh build is forced."""
    lu_dir = os.path.join(LU_MODEL_DIR, cid)
    if os.path.isdir(lu_dir):
        shutil.rmtree(lu_dir)
    readout = os.path.join(MODEL_DIR, f"model_{cid}_idea2.bin")
    if os.path.exists(readout):
        os.remove(readout)


def get_lu_model_size(cid: str) -> tuple[int, int, int]:
    """After build, look up the LU model file and load metadata."""
    cands = glob.glob(os.path.join(LU_MODEL_DIR, cid, "model_*_lu.bin"))
    if not cands:
        return (0, 0, 0)
    # use weights loader to get architecture
    try:
        from transformer_vm.model.weights import load_weights
        m, _, _ = load_weights(cands[0])
        n_layers = len(m.attn)
        d_model = m.attn[0].embed_dim
        n_params = sum(p.numel() for p in m.parameters())
        return (n_layers, d_model, n_params)
    except Exception:
        return (0, 0, 0)


def main():
    logf = open(LOG_PATH, "a")
    def log(msg):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    log("=== timing_sweep.py START ===")

    # Open CSV in append mode; write header if new
    file_is_new = not os.path.exists(OUT_CSV)
    out_f = open(OUT_CSV, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    if file_is_new:
        writer.writeheader()
        out_f.flush()

    done = already_done(OUT_CSV)
    log(f"Resuming: {len(done)} circuits already complete")

    circuits = load_dataset()
    log(f"Total circuits in dataset: {len(circuits)}")

    n_done = len(done)
    for c in circuits:
        cid = c["ID"]
        if cid in done:
            continue
        try:
            log(f"--- {cid} ({c.get('Complexity','?')}) ---")
            clear_cache(cid)
            t_total0 = time.time()
            info = build_for_circuit(cid, force=True)
            t1_b = float(info.get("lu_build_s") or 0.0)
            t1_i = float(info.get("lu_infer_s") or 0.0)
            t2_b = float(info.get("readout_build_s") or 0.0)
            t2_layers = int(info.get("n_layers") or 0)
            t2_dmodel = int(info.get("d_model") or 0)
            t2_params = int(info.get("n_params") or 0)
            t1_layers, t1_dmodel, t1_params = get_lu_model_size(cid)

            # Inference (T2)
            r0 = time.time()
            status, pred, truth, infer_s, n_tokens = run_circuit_idea2(cid, tol=0.05)
            t2_i = infer_s
            abs_err = abs(pred - truth)
            pass_50 = "PASS" if abs_err <= 0.05 else "FAIL"
            pass_75 = "PASS" if abs_err <= 0.075 else "FAIL"

            row = dict(
                circuit_id=cid, complexity=c.get("Complexity", ""),
                N=info.get("N", 0), n_free=info.get("n_free", 0),
                n_fixed=info.get("n_fixed", 0),
                T1_build_s=round(t1_b, 4), T1_infer_s=round(t1_i, 4),
                T2_build_s=round(t2_b, 4), T2_infer_s=round(t2_i, 4),
                T2_n_tokens=n_tokens,
                T1_n_layers=t1_layers, T1_d_model=t1_dmodel, T1_n_params=t1_params,
                T2_n_layers=t2_layers, T2_d_model=t2_dmodel, T2_n_params=t2_params,
                pred_v=round(pred, 4), truth_v=round(truth, 4),
                abs_err_v=round(abs_err, 4),
                pass_50mv=pass_50, pass_75mv=pass_75,
                status="OK", error="",
            )
            writer.writerow(row); out_f.flush()
            n_done += 1
            log(f"  OK  T1_build={t1_b:.2f}s T1_infer={t1_i:.2f}s T2_build={t2_b:.2f}s T2_infer={t2_i*1000:.1f}ms  err={abs_err:.4f}V  ({n_done}/{len(circuits)})")
        except Exception as e:
            tb = traceback.format_exc().replace("\n", " | ")[:500]
            log(f"  ERROR {cid}: {e}")
            row = {f: "" for f in FIELDS}
            row["circuit_id"] = cid
            row["complexity"] = c.get("Complexity", "")
            row["status"] = "ERROR"
            row["error"] = str(e)[:300]
            writer.writerow(row); out_f.flush()
            n_done += 1

    out_f.close()
    log(f"=== timing_sweep.py DONE ({n_done}/{len(circuits)}) ===")
    logf.close()


if __name__ == "__main__":
    main()
