"""E7, E8: batch-build per-circuit transformer weights.

Loops `build.build_for_circuit` over all 154 circuits and records MILP stats
plus the resulting model.bin file size. Skips circuits whose model_<ID>.bin
already exists (idempotent re-runs).

Output: `results/build_stats.csv` with one row per circuit.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
import logging
import os
import time
from contextlib import redirect_stdout

import build as build_mod
from jacobi_reference import _auto_T
from parse import parse_netlist

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = REPO_ROOT  # craft/ root — same place runner.py expects them
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_all(limit: int | None = None, skip_existing: bool = True) -> list[dict]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_csv = os.path.join(RESULTS_DIR, "build_stats.csv")

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if limit is not None:
        circuits = circuits[:limit]

    rows: list[dict] = []
    fieldnames = [
        "circuit_id", "N", "T_used", "n_layers", "d_model", "d_ffn", "n_heads",
        "vocab", "n_params", "model_size_bytes", "milp_time_s", "build_time_s",
        "skipped_existing",
    ]

    for i, c in enumerate(circuits):
        cid = c["ID"]
        out_path = os.path.join(MODEL_DIR, f"model_{cid}.bin")

        row = {fn: "" for fn in fieldnames}
        row["circuit_id"] = cid

        try:
            pc = parse_netlist(c["Netlist"])
            T = _auto_T(pc.num_nodes)
            row["N"] = pc.num_nodes
            row["T_used"] = T

            if skip_existing and os.path.exists(out_path):
                row["skipped_existing"] = True
                row["model_size_bytes"] = os.path.getsize(out_path)
                # Load to re-read shape info without rebuilding.
                try:
                    from transformer_vm.model.weights import load_weights
                    m, all_tokens, _ = load_weights(out_path)
                    row["n_layers"] = len(m.attn)
                    row["d_model"] = m.tok.weight.shape[1]
                    row["d_ffn"] = m.ff_in[0].weight.shape[0] // 2
                    row["n_heads"] = m.attn[0].num_heads
                    row["vocab"] = len(all_tokens)
                    row["n_params"] = sum(p.numel() for p in m.parameters())
                except Exception as e:
                    log.warning(f"{cid}: could not reread existing model: {e}")
            else:
                # Build fresh — capture stdout noise from MILP solver.
                t0 = time.time()
                devnull = open(os.devnull, "w")
                try:
                    with redirect_stdout(devnull):
                        result = build_mod.build_for_circuit(
                            cid, T=T, out_path=out_path, plan_only=False
                        )
                finally:
                    devnull.close()
                build_time = time.time() - t0

                row["n_layers"] = result["n_layers"]
                row["d_model"] = result["d_model"]
                row["d_ffn"] = result["d_ffn"]
                row["vocab"] = result["vocab"]
                row["n_params"] = result["n_params"]
                row["model_size_bytes"] = os.path.getsize(out_path)
                row["build_time_s"] = build_time
                row["skipped_existing"] = False

            print(
                f"[{i+1:>3}/{len(circuits)}] {cid:<10} "
                f"N={row['N']:<3} T={row['T_used']:<5} "
                f"layers={row['n_layers']} d_model={row['d_model']} "
                f"size={row['model_size_bytes']}B "
                f"{'(cached)' if row['skipped_existing'] else ''}",
                flush=True,
            )

        except Exception as e:
            log.error(f"{cid}: build failed: {e}")
            row["error"] = str(e)

        rows.append(row)

        # Incremental checkpoint
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"\n[build_all] wrote {stats_csv} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_all(limit=limit)
