"""
T1.4 — Tolerance sweep: pass rate vs tolerance for all 4 compiled algorithms.
Reads existing per-circuit error CSVs (no recompilation).
Outputs: fig_tol_sweep.pdf (for tmlr/figures/) and summary stats to stdout.
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CADJ_DIR = os.path.join(BASE, "craft", "cadj_results")
RAMBAAN_DIR = os.path.join(BASE, "lu_pipeline", "results")
OUT_FIG = os.path.join(BASE, "tmlr", "figures", "fig_tol_sweep.pdf")

# ── load per-circuit errors for each algorithm ─────────────────────────────

def _load_err(path, col="abs_error"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["circuit_id"]: float(r[col]) for r in rows if r.get(col, "").strip()}

# Confirmed mapping (verified against paper pass-rate numbers):
#   LU-direct  → lu_pipeline results_main_lu_direct (150/154 PASS)
#   iJacobi    → cadj_results/results_idea2     (150/154 PASS, updated from paper's 135)
#   RBSOR      → cadj_results/results_main_cadj_v01  (123/154 PASS, matches paper)
#   Disc.Jacobi→ cadj_results/results_main_cadj (96/154 PASS, close to paper's 90)
lu      = _load_err(os.path.join(RAMBAAN_DIR, "results_main_lu_direct.csv"))
ijacob  = _load_err(os.path.join(CADJ_DIR, "results_idea2.csv"))
rbsor   = _load_err(os.path.join(CADJ_DIR, "results_main_cadj_v01.csv"))
djacob  = _load_err(os.path.join(CADJ_DIR, "results_main_cadj.csv"))

# canonicalize to same 154-circuit key set
common = sorted(set(lu) & set(ijacob) & set(rbsor) & set(djacob))
print(f"Common circuits: {len(common)}")

algorithms = {
    "LU-direct":      [lu[c]     for c in common],
    "iJacobi":        [ijacob[c] for c in common],
    "Red-Black SOR":  [rbsor[c]  for c in common],
    "Discrete Jacobi":[djacob[c] for c in common],
}

N = len(common)
tolerances_mV = list(range(5, 105, 5))   # 5 mV to 100 mV

# ── compute pass rates ─────────────────────────────────────────────────────

def pass_rate(errs, tol_mV):
    tol = tol_mV / 1000.0
    return sum(1 for e in errs if e <= tol) / len(errs)

rates = {
    algo: [pass_rate(errs, t) for t in tolerances_mV]
    for algo, errs in algorithms.items()
}

# ── print summary table ────────────────────────────────────────────────────

print(f"\n{'Tolerance':>10}  {'LU-direct':>10}  {'iJacobi':>10}  {'RBSOR':>10}  {'Jacobi':>10}")
for t in [10, 20, 30, 40, 50, 60, 75, 100]:
    idx = tolerances_mV.index(t)
    row = {a: rates[a][idx] for a in algorithms}
    print(
        f"{t:>9}mV  "
        f"{row['LU-direct']:>9.1%}  "
        f"{row['iJacobi']:>9.1%}  "
        f"{row['Red-Black SOR']:>9.1%}  "
        f"{row['Discrete Jacobi']:>9.1%}"
    )

# 50 mV summary for paper
idx50 = tolerances_mV.index(50)
print("\n--- At 50 mV (paper headline) ---")
for algo, errs in algorithms.items():
    n_pass = sum(1 for e in errs if e <= 0.050)
    print(f"  {algo}: {n_pass}/{N} = {n_pass/N*100:.1f}%")

# ── figure ─────────────────────────────────────────────────────────────────

COLORS = {
    "LU-direct":      "#1f77b4",
    "iJacobi":        "#2ca02c",
    "Red-Black SOR":  "#ff7f0e",
    "Discrete Jacobi":"#d62728",
}
STYLES = {
    "LU-direct":      "-",
    "iJacobi":        "--",
    "Red-Black SOR":  "-.",
    "Discrete Jacobi":":",
}

fig, ax = plt.subplots(figsize=(5.5, 3.5))

for algo in ["LU-direct", "iJacobi", "Red-Black SOR", "Discrete Jacobi"]:
    ax.plot(
        tolerances_mV, [r * 100 for r in rates[algo]],
        linestyle=STYLES[algo], color=COLORS[algo],
        linewidth=1.8, label=algo,
    )

ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
           label="50 mV (1 grid step)")
ax.set_xlabel("Tolerance (mV)", fontsize=10)
ax.set_ylabel("Pass rate (%)", fontsize=10)
ax.set_ylim(-2, 102)
ax.set_xlim(5, 100)
ax.set_xticks([10, 20, 30, 40, 50, 60, 75, 100])
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
print(f"\nSaved: {OUT_FIG}")
