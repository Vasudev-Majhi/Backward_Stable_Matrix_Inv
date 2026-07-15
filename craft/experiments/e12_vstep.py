"""E12 — V_STEP ablation on Mode B (quantization-limited) failures.

Mode B = circuits that fully converge (Jacobi ref matches truth) but land in
the wrong 0.05V bin due to coarse quantization. Hypothesis: reducing V_STEP
(smaller quantization step, larger vocab) should fix them without changing
the algorithm.

We run 10 Mode B circuits at V_STEP ∈ {0.05, 0.025, 0.01, 0.005} — the
standard 0.05V step, plus three finer alternatives. Each combination keeps
K_LEVELS·V_STEP/SCALE ≈ 24V (the full voltage range).

Expected result: most or all 10 circuits pass at V_STEP=0.025 or finer,
proving the quantization floor is a tunable design parameter, not a
fundamental limit.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("E12")

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MODE_B_CIRCUITS = [
    "CKT_0068", "CKT_0098", "CKT_0032", "CKT_0035", "CKT_0015",
    "CKT_0105", "CKT_0067", "CKT_0048", "CKT_0094", "CKT_0092",
]

# (V_STEP_volts, K_LEVELS). Keep K_LEVELS * V_STEP_volts >= 24V.
V_STEP_VARIANTS = [
    (0.05,  480),
    (0.025, 960),
    (0.01,  2400),
    (0.005, 4800),
]


def run_e12(limit_ids: list[str] | None = None) -> list[dict]:
    from dsl_runner import run_one_dsl
    from jacobi_reference import _auto_T
    from parse import parse_netlist

    circuits_wanted = limit_ids if limit_ids is not None else MODE_B_CIRCUITS

    # Load all relevant circuits from the dataset.
    circuits: dict[str, dict] = {}
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] in circuits_wanted:
                circuits[c["ID"]] = c

    rows: list[dict] = []

    for cid in circuits_wanted:
        if cid not in circuits:
            log.warning(f"{cid}: not in dataset, skipping")
            continue
        c = circuits[cid]
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = float(c["Ground_Truth_Vout"])
        T = _auto_T(pc.num_nodes)

        for v_step_volts, k_levels in V_STEP_VARIANTS:
            v_step_scaled = int(round(v_step_volts * 10000))
            t0 = time.time()
            try:
                r = run_one_dsl(
                    c["Netlist"], target, T, parsed=pc,
                    v_step=v_step_scaled, k_levels=k_levels,
                )
                err = abs(r["pred_V"] - truth)
                ok = err <= 0.05
                row = {
                    "circuit_id": cid,
                    "N": pc.num_nodes,
                    "T": T,
                    "V_STEP_V": v_step_volts,
                    "K_LEVELS": k_levels,
                    "pred_V": f"{r['pred_V']:.4f}",
                    "truth_V": f"{truth:.4f}",
                    "abs_error": f"{err:.4f}",
                    "pass": ok,
                    "seq_length": r["seq_length"],
                    "runtime_s": f"{time.time() - t0:.2f}",
                    "error": "",
                }
                log.info(
                    f"{cid} V_STEP={v_step_volts}V: pred={r['pred_V']:.4f} "
                    f"truth={truth:.4f} err={err:.4f} {'PASS' if ok else 'FAIL'}"
                )
            except Exception as e:  # noqa: BLE001
                log.error(f"{cid} V_STEP={v_step_volts}V: {e}")
                row = {
                    "circuit_id": cid, "V_STEP_V": v_step_volts,
                    "K_LEVELS": k_levels, "error": f"{type(e).__name__}: {e}",
                }
            rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "vstep_ablation_E12.csv")
    fieldnames = [
        "circuit_id", "N", "T", "V_STEP_V", "K_LEVELS",
        "pred_V", "truth_V", "abs_error", "pass",
        "seq_length", "runtime_s", "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {out_csv} ({len(rows)} rows)")

    # Pass-rate summary per V_STEP
    by_vstep: dict[float, list[bool]] = {}
    for r in rows:
        v = r.get("V_STEP_V")
        if v is None or "error" in r and r["error"]:
            continue
        by_vstep.setdefault(v, []).append(bool(r["pass"]))
    log.info("Pass rates by V_STEP:")
    for v in sorted(by_vstep):
        bs = by_vstep[v]
        log.info(f"  V_STEP={v}V: {sum(bs)}/{len(bs)} pass")

    return rows


if __name__ == "__main__":
    run_e12()
