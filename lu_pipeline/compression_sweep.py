"""Compression sweep on the in-house 154 circuits.

For each circuit's compiled CRAFT LU-direct model, apply each of {float64
baseline, int16, int8, int4, prune-10%, prune-50%, prune-90%} compression
settings to the weights, run inference, and record:

  - pred_v: prediction with this setting
  - abs_error_v: |pred_v - truth_v|
  - delta_vs_float64: |pred_v - pred_v(float64)|
  - pass_50mv: PASS/FAIL
  - pass_75mv: PASS/FAIL

Output: results/compression_154.csv with one row per (circuit_id, setting).

This is the experiment that backs the compression-asymmetry figure (Figure 5
in the paper): pruning is bit-identical, quantization is catastrophic, across
all 154 circuits.
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import copy
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout

import torch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")

# Names of compression settings (deterministic order)
SETTINGS = [
    "float64",
    "int16", "int8", "int4",
    "prune10", "prune50", "prune90",
]

FIELDS = [
    "circuit_id", "setting",
    "N", "n_free",
    "pred_v", "truth_v", "abs_error_v",
    "delta_vs_float64", "pass_50mv", "pass_75mv",
    "build_status", "run_status", "run_time_s",
    "error",
]


# ---------------------------------------------------------------------------
# Compression operators (operate on cloned model; original untouched)
# ---------------------------------------------------------------------------

def _quantize_inplace(model: torch.nn.Module, n_bits: int) -> None:
    """Symmetric per-tensor quantize+dequantize. Simulates int_n weight storage."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point and p.numel() > 0:
                w = p.data
                amax = w.abs().max().item()
                if amax == 0.0:
                    continue
                scale = amax / qmax
                w_q = torch.round(w / scale).clamp(qmin, qmax) * scale
                p.data.copy_(w_q.to(p.dtype))


def _prune_inplace(model: torch.nn.Module, sparsity: float) -> None:
    """Magnitude prune: zero-out the smallest |weight| up to (sparsity) fraction
    of each tensor. Per-tensor threshold."""
    assert 0.0 <= sparsity <= 1.0
    if sparsity == 0.0:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point and p.numel() > 0:
                w = p.data
                k = max(1, int(round(sparsity * w.numel())))
                if k >= w.numel():
                    p.data.zero_()
                    continue
                # threshold = k-th smallest magnitude
                flat = w.abs().reshape(-1)
                threshold = torch.kthvalue(flat, k).values.item()
                mask = w.abs() > threshold
                p.data.mul_(mask.to(p.dtype))


def _apply_setting(model: torch.nn.Module, setting: str) -> None:
    if setting == "float64":
        return
    if setting == "int16":
        _quantize_inplace(model, 16); return
    if setting == "int8":
        _quantize_inplace(model, 8); return
    if setting == "int4":
        _quantize_inplace(model, 4); return
    if setting == "prune10":
        _prune_inplace(model, 0.10); return
    if setting == "prune50":
        _prune_inplace(model, 0.50); return
    if setting == "prune90":
        _prune_inplace(model, 0.90); return
    raise ValueError(f"unknown setting: {setting!r}")


# ---------------------------------------------------------------------------
# Per-circuit runner: load once, apply each setting on a clone, run inference
# ---------------------------------------------------------------------------

def _run_one_circuit(cid: str, model_path: str, netlist: str, truth_v: float,
                      tol_50: float = 0.05, tol_75: float = 0.075) -> list[dict]:
    """Returns a list of result dicts (one per setting)."""
    from transformer_vm.model.weights import load_weights, save_weights  # type: ignore
    from runner_lu_direct import run_one  # type: ignore

    rows: list[dict] = []
    pred_float64: float | None = None

    # Load the clean model once; we'll re-load (for clean weights) per setting
    # via state_dict snapshotting to avoid re-reading the disk every time.
    model0, all_tokens, _tok_to_idx = load_weights(model_path)
    sd_clean = copy.deepcopy(model0.state_dict())
    n_free = 0  # filled below from sidecar
    sidecar_path = model_path + ".slots.json"
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            sd_meta = json.load(f)
        n_free = int(sd_meta.get("n_free", 0))
    else:
        return [{
            "circuit_id": cid, "setting": "_no_sidecar",
            "error": "missing sidecar JSON",
        }]

    tmp_dir = tempfile.mkdtemp(prefix=f"compress_{cid}_")
    sidecar_dst = os.path.join(tmp_dir, "model.bin.slots.json")
    shutil.copy(sidecar_path, sidecar_dst)
    tmp_model_path = os.path.join(tmp_dir, "model.bin")

    try:
        for setting in SETTINGS:
            row = {fn: "" for fn in FIELDS}
            row["circuit_id"] = cid
            row["setting"] = setting
            row["truth_v"] = f"{truth_v:.4f}"
            row["n_free"] = n_free

            try:
                # Restore clean weights into model0
                model0.load_state_dict(sd_clean)
                _apply_setting(model0, setting)

                with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                    save_weights(model0, all_tokens, tmp_model_path)

                t0 = time.time()
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                    status, pred_v, _truth, elapsed = run_one(
                        model_path=tmp_model_path, netlist=netlist,
                        truth=truth_v, tol=tol_50, v_step=None, k_levels=None,
                        verbose=False, use_hull=False,
                    )
                dt = time.time() - t0

                row["pred_v"] = f"{pred_v:.6f}"
                err = abs(pred_v - truth_v)
                row["abs_error_v"] = f"{err:.6f}"
                row["pass_50mv"] = "PASS" if err <= tol_50 else "FAIL"
                row["pass_75mv"] = "PASS" if err <= tol_75 else "FAIL"
                row["run_status"] = status
                row["run_time_s"] = f"{dt:.2f}"

                if setting == "float64":
                    pred_float64 = pred_v
                    row["delta_vs_float64"] = "0.000000"
                elif pred_float64 is not None:
                    row["delta_vs_float64"] = f"{abs(pred_v - pred_float64):.6f}"
            except Exception as e:
                row["run_status"] = "error"
                row["error"] = repr(e)

            rows.append(row)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default=os.environ.get(
                        "CRAFT_DATASET",
                        os.path.join(PROJECT, "nohull_nouniversal_jacob",
                                       "dataset", "circuit_dataset_rv.jsonl"),
                    ))
    ap.add_argument("--model-suffix", type=str, default="_v2",
                    help="suffix on model file: model_<cid>_lu_direct{suffix}.bin")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default=os.path.join(RESULTS_DIR, "compression_154.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    circuits: list[dict] = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.start:
        circuits = circuits[args.start:]
    if args.limit is not None:
        circuits = circuits[: args.limit]

    print(f"[compression] {len(circuits)} circuits, {len(SETTINGS)} settings = "
          f"{len(circuits) * len(SETTINGS)} runs", flush=True)

    out_csv = args.out
    all_rows: list[dict] = []
    if os.path.exists(out_csv):
        # resume: read existing rows, skip circuits that already have all settings
        with open(out_csv, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                all_rows.append({k: row.get(k, "") for k in FIELDS})
        done_cids = {row["circuit_id"] for row in all_rows
                     if sum(1 for r2 in all_rows if r2["circuit_id"] == row["circuit_id"]) >= len(SETTINGS)}
        circuits = [c for c in circuits if c["ID"] not in done_cids]
        print(f"[compression] resuming; {len(done_cids)} circuits already done, "
              f"{len(circuits)} remaining", flush=True)

    def _flush():
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

    pass50_count = {s: 0 for s in SETTINGS}
    pass75_count = {s: 0 for s in SETTINGS}
    fail50_count = {s: 0 for s in SETTINGS}

    for i, c in enumerate(circuits):
        cid = c["ID"]
        truth_v = float(c["Ground_Truth_Vout"])
        netlist = c["Netlist"]
        model_path = os.path.join(HERE, f"model_{cid}_lu_direct{args.model_suffix}.bin")

        if not os.path.exists(model_path):
            row = {fn: "" for fn in FIELDS}
            row["circuit_id"] = cid
            row["setting"] = "_no_model"
            row["error"] = f"missing {model_path}"
            all_rows.append(row)
            print(f"[{i+1:>3}/{len(circuits)}] {cid:<12} NO MODEL", flush=True)
            _flush()
            continue

        t0 = time.time()
        try:
            rows = _run_one_circuit(cid, model_path, netlist, truth_v)
            all_rows.extend(rows)
            for r in rows:
                if r["pass_50mv"] == "PASS":
                    pass50_count[r["setting"]] += 1
                elif r["pass_50mv"] == "FAIL":
                    fail50_count[r["setting"]] += 1
                if r["pass_75mv"] == "PASS":
                    pass75_count[r["setting"]] += 1
            _flush()
            dt = time.time() - t0
            tag = " ".join(
                f"{s}={'P' if r['pass_50mv']=='PASS' else 'F' if r['pass_50mv']=='FAIL' else '!'}"
                for s, r in zip(SETTINGS, rows)
            )
            print(f"[{i+1:>3}/{len(circuits)}] {cid:<12} {tag}  {dt:5.1f}s",
                  flush=True)
        except Exception as e:
            print(f"[{i+1:>3}/{len(circuits)}] {cid} EXC: {e}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)

    print("\n=== COMPRESSION SUMMARY ===")
    print(f"{'setting':<12} {'pass@50mV':>10} {'pass@75mV':>10}")
    for s in SETTINGS:
        n_total = pass50_count[s] + fail50_count[s]
        p50 = pass50_count[s]
        p75 = pass75_count[s]
        pct = p50 / n_total * 100 if n_total else 0.0
        print(f"{s:<12} {p50:>5}/{n_total:<4} ({pct:5.1f}%) {p75:>5}/{n_total:<4}")

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
