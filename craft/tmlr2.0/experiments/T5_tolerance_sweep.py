"""Reviewer point #5: LU-direct tolerance sensitivity sweep.

Recomputes pass rates at multiple voltage tolerances (25, 50, 75, 100, 150 mV)
from the existing per-circuit prediction CSV. Demonstrates that the 4 boundary
failures at 50mV are quantization-grid near-misses (they vanish at 75mV+).

No new builds or inferences — pure CSV post-processing.

Output: results/T5_tol/lu_tolerance_sweep.csv
"""
from __future__ import annotations
import os, csv

HOME = os.path.expanduser("~")
INPUT_CSV = os.path.join(HOME, "craft_release", "lu_pipeline", "results",
                        "results_main_lu_direct_v2.csv")
OUT_DIR = os.path.join(HOME, "craft_release", "tmlr2_runs", "results", "T5_tol")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tolerances_V = [0.025, 0.050, 0.075, 0.100, 0.150]

    rows = list(csv.DictReader(open(INPUT_CSV)))
    print(f"loaded {len(rows)} circuit rows from {INPUT_CSV}")

    # Per-tolerance pass rate
    fields_summary = ["tolerance_mV", "pass_total", "pass_count", "pass_rate"]
    fields_summary += ["pass_Basic", "pass_Intermediate", "pass_Hard"]
    summary = []

    tiers = ["Basic", "Intermediate", "Hard"]
    for tol_V in tolerances_V:
        rec = {"tolerance_mV": int(tol_V * 1000)}
        passed = 0
        per_tier = {t: 0 for t in tiers}
        tier_totals = {t: 0 for t in tiers}
        for r in rows:
            try:
                err = float(r["abs_error"])
            except Exception:
                continue
            tier = r["complexity"]
            tier_totals[tier] = tier_totals.get(tier, 0) + 1
            # Paper uses strict less-than at the labelled threshold:
            # 50 mV pass = err < 0.050, so a circuit at exactly 50.00 mV FAILS.
            if err < tol_V:
                passed += 1
                per_tier[tier] = per_tier.get(tier, 0) + 1
        rec["pass_total"] = len(rows)
        rec["pass_count"] = passed
        rec["pass_rate"] = round(passed / len(rows), 4)
        for t in tiers:
            rec[f"pass_{t}"] = f"{per_tier.get(t,0)}/{tier_totals.get(t,0)}"
        summary.append(rec)
        print(f"  tol = {int(tol_V*1000):>3d} mV  pass = {passed:>3d}/{len(rows)} "
              f"({100*passed/len(rows):>5.1f}%)  Basic={per_tier['Basic']}/{tier_totals['Basic']}  "
              f"Inter={per_tier['Intermediate']}/{tier_totals['Intermediate']}  "
              f"Hard={per_tier['Hard']}/{tier_totals['Hard']}")

    out_csv = os.path.join(OUT_DIR, "lu_tolerance_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_summary); w.writeheader()
        for r in summary: w.writerow(r)
    print(f"\nwrote {out_csv}")

    # Also identify the boundary failures at 50 mV and report their errors
    # (strict less-than: err >= 0.050 fails)
    boundary = sorted(
        [(r["circuit_id"], float(r["abs_error"]))
         for r in rows if float(r["abs_error"]) >= 0.050],
        key=lambda t: t[1])
    print(f"\nCircuits failing 50 mV (n={len(boundary)}):")
    for cid, e in boundary:
        print(f"  {cid}: err = {e*1000:.1f} mV")

    out_fail = os.path.join(OUT_DIR, "boundary_failures.csv")
    with open(out_fail, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["circuit", "abs_error_V", "abs_error_mV"])
        for cid, e in boundary: w.writerow([cid, e, round(e*1000, 2)])
    print(f"wrote {out_fail}")


if __name__ == "__main__":
    main()
