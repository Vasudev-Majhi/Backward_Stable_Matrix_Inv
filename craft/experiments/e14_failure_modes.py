"""E14 — Failure mode classification.

Reads results_main.csv and classifies every FAIL into one of:
  Mode A — Jacobi reference itself is wrong (ρ≈1 non-convergence)
  Mode B-simple — quantization rounding miss (|err_vs_jacobi| ~ V_STEP)
  Mode B-compound — quantization bias compounded at high ρ
  Mode C — boundary/edge case

Outputs failure_modes_E14.csv (one row per FAIL with a mode label) and
a pass/fail-mode breakdown in summary.md.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("E14")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Thresholds
JACOBI_WRONG_THRESHOLD_V = 0.05   # |jacobi - truth| > this → Jacobi itself fails (Mode A)
QUANTIZATION_STEP_V = 0.05        # one V_STEP bin
B_SIMPLE_MULTIPLIER = 2.0         # err ≤ 2*V_STEP → B-simple; else B-compound


def classify(row: dict) -> tuple[str, dict]:
    """Return (mode_label, diagnostic_fields) for one results_main.csv row."""
    out = {"circuit_id": row["circuit_id"],
           "complexity": row["complexity"],
           "N": row["N"],
           "rho_M": row["rho_M"],
           "kappa_A": row["kappa_A"],
           "truth_V": row["truth_V"],
           "pred_V_dsl": row["pred_V_dsl"],
           "jacobi_ref_V": row["jacobi_ref_V"],
           "abs_error": row["abs_error"]}

    if row["pass_fail"] != "FAIL":
        out["mode"] = "PASS"
        return "PASS", out

    try:
        pred = float(row["pred_V_dsl"])
        truth = float(row["truth_V"])
        jref = float(row["jacobi_ref_V"]) if row["jacobi_ref_V"] else None
        rho = float(row["rho_M"]) if row["rho_M"] else 0.0
    except (ValueError, TypeError):
        out["mode"] = "UNPARSEABLE"
        return "UNPARSEABLE", out

    if jref is None:
        out["mode"] = "UNPARSEABLE"
        return "UNPARSEABLE", out

    jacobi_err = abs(jref - truth)
    dsl_vs_jacobi = abs(pred - jref)
    dsl_vs_truth = abs(pred - truth)

    out["jacobi_err"] = f"{jacobi_err:.4f}"
    out["dsl_vs_jacobi"] = f"{dsl_vs_jacobi:.4f}"

    # Mode A: Jacobi itself is wrong (the algorithm can't reach truth)
    if jacobi_err > JACOBI_WRONG_THRESHOLD_V:
        mode = "A_non_convergent"
    # Mode C: DSL very close to pass tolerance (floating-point epsilon boundary)
    elif dsl_vs_truth <= QUANTIZATION_STEP_V + 1e-6:
        mode = "C_boundary"
    # Mode B-simple: quantization miss within ~2 V_STEPs
    elif dsl_vs_jacobi <= B_SIMPLE_MULTIPLIER * QUANTIZATION_STEP_V:
        mode = "B_simple"
    # Mode B-compound: DSL fixed point drifted far from Jacobi's
    else:
        mode = "B_compound"

    out["mode"] = mode
    return mode, out


def run_e14() -> dict:
    main_csv = os.path.join(RESULTS_DIR, "results_main.csv")
    if not os.path.exists(main_csv):
        log.error(f"{main_csv} not found; run Phase B first")
        return {}

    with open(main_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classified: list[dict] = []
    mode_counts: dict[str, int] = {}
    for r in rows:
        mode, info = classify(r)
        classified.append(info)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    # Write full per-circuit classification
    out_csv = os.path.join(RESULTS_DIR, "failure_modes_E14.csv")
    fieldnames = [
        "circuit_id", "complexity", "N", "rho_M", "kappa_A",
        "truth_V", "pred_V_dsl", "jacobi_ref_V", "abs_error",
        "jacobi_err", "dsl_vs_jacobi", "mode",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(classified)
    log.info(f"wrote {out_csv} ({len(classified)} rows)")

    log.info("Mode breakdown:")
    for m, c in sorted(mode_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {m:<20} {c}")

    return {"mode_counts": mode_counts, "rows": classified}


if __name__ == "__main__":
    run_e14()
