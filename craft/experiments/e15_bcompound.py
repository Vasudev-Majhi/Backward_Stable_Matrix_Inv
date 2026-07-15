"""E15 — B-compound V_STEP test.

Does finer V_STEP fix B-compound failures (cases where the DSL drifts far
from Jacobi reference at high ρ)?

3 representative B-compound circuits × 4 V_STEP values:
  CKT_0146 (ρ=0.971, drift=5.70V)
  CKT_0154 (ρ=0.956, drift=2.29V)
  CKT_0043 (ρ=0.991, drift=2.04V)
  V_STEP ∈ {0.05, 0.01, 0.005, 0.001}

Outcomes:
  - All pass at 0.001V  → precision-fixable; re-run E1 at that V_STEP
  - Still fail at 0.001V → B-compound is structural, not precision-fixable
                           (DSL's own fixed point differs from Jacobi's at high ρ)
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
log = logging.getLogger("E15")

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

B_COMPOUND_CIRCUITS = ["CKT_0146", "CKT_0154", "CKT_0043"]

# (V_STEP_volts, K_LEVELS). Each keeps K*V >= 24V.
V_STEP_VARIANTS = [
    (0.05,  480),
    (0.01,  2400),
    (0.005, 4800),
    (0.001, 24000),
]


def run_e15() -> list[dict]:
    from dsl_runner import run_one_dsl
    from jacobi_reference import _auto_T, jacobi_solve_parsed
    from parse import parse_netlist
    from spectrum import compute_rho_kappa

    circuits: dict[str, dict] = {}
    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] in B_COMPOUND_CIRCUITS:
                circuits[c["ID"]] = c

    rows: list[dict] = []
    for cid in B_COMPOUND_CIRCUITS:
        if cid not in circuits:
            log.warning(f"{cid}: not in dataset")
            continue
        c = circuits[cid]
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = float(c["Ground_Truth_Vout"])
        T = _auto_T(pc.num_nodes)
        rho, _ = compute_rho_kappa(pc)
        jref = jacobi_solve_parsed(pc, target, T)

        for v_step_volts, k_levels in V_STEP_VARIANTS:
            v_step_scaled = int(round(v_step_volts * 10000))
            t0 = time.time()
            try:
                r = run_one_dsl(
                    c["Netlist"], target, T, parsed=pc,
                    v_step=v_step_scaled, k_levels=k_levels,
                )
                err_vs_truth = abs(r["pred_V"] - truth)
                err_vs_jacobi = abs(r["pred_V"] - jref)
                ok = err_vs_truth <= 0.05
                row = {
                    "circuit_id": cid,
                    "N": pc.num_nodes,
                    "rho_M": f"{rho:.4f}",
                    "T": T,
                    "V_STEP_V": v_step_volts,
                    "K_LEVELS": k_levels,
                    "pred_V": f"{r['pred_V']:.6f}",
                    "truth_V": f"{truth:.6f}",
                    "jacobi_ref_V": f"{jref:.6f}",
                    "err_vs_truth": f"{err_vs_truth:.6f}",
                    "err_vs_jacobi": f"{err_vs_jacobi:.6f}",
                    "pass": ok,
                    "seq_length": r["seq_length"],
                    "runtime_s": f"{time.time() - t0:.2f}",
                    "error": "",
                }
                log.info(
                    f"{cid} V_STEP={v_step_volts}V: pred={r['pred_V']:.4f} "
                    f"jref={jref:.4f} truth={truth:.4f} "
                    f"err_vs_truth={err_vs_truth:.4f} {'PASS' if ok else 'FAIL'}"
                )
            except Exception as e:  # noqa: BLE001
                log.error(f"{cid} V_STEP={v_step_volts}V: {e}")
                row = {
                    "circuit_id": cid, "V_STEP_V": v_step_volts,
                    "error": f"{type(e).__name__}: {e}",
                }
            rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "bcompound_vstep_E15.csv")
    fieldnames = [
        "circuit_id", "N", "rho_M", "T", "V_STEP_V", "K_LEVELS",
        "pred_V", "truth_V", "jacobi_ref_V", "err_vs_truth", "err_vs_jacobi",
        "pass", "seq_length", "runtime_s", "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {out_csv} ({len(rows)} rows)")

    # Decision signal: at V_STEP=0.01, how many of the 3 pass?
    from collections import defaultdict
    pass_by_vstep: dict[float, int] = defaultdict(int)
    total_by_vstep: dict[float, int] = defaultdict(int)
    for r in rows:
        v = r.get("V_STEP_V")
        if v is None or r.get("error"):
            continue
        total_by_vstep[v] += 1
        if r["pass"]:
            pass_by_vstep[v] += 1
    decision_csv = os.path.join(RESULTS_DIR, "e15_decision.txt")
    with open(decision_csv, "w") as f:
        for v in sorted(total_by_vstep, reverse=True):
            msg = f"V_STEP={v}V: {pass_by_vstep[v]}/{total_by_vstep[v]}"
            log.info(msg)
            f.write(msg + "\n")
        # Pick the LARGEST V_STEP that passes all 3
        best_v = 0.05
        for v in sorted(total_by_vstep, reverse=True):
            if pass_by_vstep[v] == total_by_vstep[v]:
                best_v = v
        f.write(f"RECOMMENDED_V_STEP={best_v}\n")
        log.info(f"recommended V_STEP for E16: {best_v}V")
    return rows


if __name__ == "__main__":
    run_e15()
