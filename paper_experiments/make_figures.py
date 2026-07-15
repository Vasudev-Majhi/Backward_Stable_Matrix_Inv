"""
NeurIPS 2026 — CRAFT paper figure generation.
Reads all result CSVs and produces publication-quality figures.
Run from ./
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── NeurIPS style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "results")
FIGS   = os.path.join(BASE, "figures")
LOCAL  = "results"
os.makedirs(FIGS, exist_ok=True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")

# ── Colour palette (colour-blind friendly) ───────────────────────────────────
CMAP = {
    "PASS":            "#2196F3",   # blue
    "B_simple":        "#FF9800",   # orange
    "B_compound":      "#F44336",   # red
    "A_non_convergent":"#9C27B0",   # purple
    "C_boundary":      "#4CAF50",   # green
    "UNKNOWN":         "#9E9E9E",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
fm   = pd.read_csv(os.path.join(LOCAL, "failure_modes_E14.csv"))
ts   = pd.read_csv(os.path.join(RES,   "timing_sweep.csv"))
th1  = pd.read_csv(os.path.join(RES,   "theorem1_verification.csv"))
haT1 = pd.read_csv(os.path.join(RES,   "head_ablation_T1.csv"))
haT2 = pd.read_csv(os.path.join(RES,   "head_ablation_T2.csv"))
cmT1 = pd.read_csv(os.path.join(RES,   "compression_T1.csv"))
cmT2 = pd.read_csv(os.path.join(RES,   "compression_T2.csv"))
rpT1 = pd.read_csv(os.path.join(RES,   "residual_probing_T1.csv"))
rpT2 = pd.read_csv(os.path.join(RES,   "residual_probing_T2.csv"))
apT1 = pd.read_csv(os.path.join(RES,   "attention_patterns_T1.csv"))

# Normalise failure mode column name
fm = fm.rename(columns={"mode": "failure_mode"})
fm_map = dict(zip(fm["circuit_id"], fm["failure_mode"]))

# Merge failure mode into timing sweep
ts["failure_mode"] = ts["circuit_id"].map(fm_map).fillna("UNKNOWN")
ts_ok = ts[ts["status"] == "OK"]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Failure Mode Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: failure mode distribution")
mode_order = ["PASS", "B_simple", "B_compound", "A_non_convergent", "C_boundary"]
labels_nice = {
    "PASS": "PASS",
    "B_simple": r"$B_{\mathrm{simple}}$",
    "B_compound": r"$B_{\mathrm{compound}}$",
    "A_non_convergent": r"$A_{\mathrm{non\text{-}conv}}$",
    "C_boundary": r"$C_{\mathrm{boundary}}$",
}

counts = fm["failure_mode"].value_counts()
# also get rho distribution per mode
fm2 = fm.copy()
fm2["rho_M"] = fm2["circuit_id"].map(dict(zip(th1["circuit_id"], th1["rho_M"])))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# Left: bar chart
ax = axes[0]
vals = [counts.get(m, 0) for m in mode_order]
bars = ax.bar(range(len(mode_order)), vals,
              color=[CMAP[m] for m in mode_order], width=0.6, edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            str(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(range(len(mode_order)))
ax.set_xticklabels([labels_nice[m] for m in mode_order], fontsize=7.5)
ax.set_ylabel("Circuit count")
ax.set_title("(a) Failure mode distribution ($n=154$)", loc="left", fontsize=9)
ax.set_ylim(0, max(vals) * 1.18)

# Right: ρ(M) strip plot per mode
ax = axes[1]
for i, mode in enumerate(mode_order):
    sub = fm2[fm2["failure_mode"] == mode]["rho_M"].dropna()
    ax.scatter(sub, [i] * len(sub), c=CMAP[mode], alpha=0.55, s=14, edgecolors="none")
ax.set_yticks(range(len(mode_order)))
ax.set_yticklabels([labels_nice[m] for m in mode_order], fontsize=7.5)
ax.set_xlabel(r"Spectral radius $\rho(M)$")
ax.set_title(r"(b) $\rho(M)$ by failure mode", loc="left", fontsize=9)
ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_xlim(-0.02, 1.08)

fig.tight_layout(w_pad=3)
save(fig, "fig1_failure_modes")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Theorem 1 Empirical Verification
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: Theorem 1 verification")

th = th1.copy()
th["failure_mode"] = th["circuit_id"].map(fm_map).fillna("UNKNOWN")
th["abs_err"] = th["circuit_id"].map(dict(zip(fm["circuit_id"], fm["abs_error"]))).fillna(np.nan)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

# (a) ρ(M) vs absolute error
ax = axes[0]
for mode in mode_order:
    sub = th[th["failure_mode"] == mode]
    ax.scatter(sub["rho_M"], sub["abs_err"], c=CMAP[mode], alpha=0.65,
               s=12, label=labels_nice[mode], edgecolors="none")
ax.axhline(0.05, color="gray", lw=0.9, ls="--", label="50 mV tol")
ax.axhline(0.075, color="gray", lw=0.9, ls=":", label="75 mV tol")
ax.set_xlabel(r"$\rho(M)$")
ax.set_ylabel("Absolute error (V)")
ax.set_title(r"(a) $\rho(M)$ vs error", loc="left", fontsize=9)
ax.set_yscale("log")
ax.set_ylim(1e-4, 30)

# (b) Stall radius vs abs_err for B_compound
ax = axes[1]
bc = th[th["failure_mode"] == "B_compound"].copy()
bc["r_stall_finite"] = bc["r_stall"].replace([np.inf, -np.inf], np.nan)
non_bc = th[th["failure_mode"] != "B_compound"]
ax.scatter(non_bc["r_stall"].clip(0, 5), non_bc["abs_err"],
           c=[CMAP[m] for m in non_bc["failure_mode"]], alpha=0.3, s=10, edgecolors="none")
ax.scatter(bc["r_stall_finite"], bc["abs_err"],
           c=CMAP["B_compound"], alpha=0.85, s=22, edgecolors="black", linewidths=0.4,
           label=r"$B_{\mathrm{compound}}$", zorder=5)
ax.axvline(0.05, color="black", lw=0.9, ls="--", alpha=0.7, label=r"$r_{\mathrm{stall}}=\mathrm{tol}$")
ax.set_xlabel(r"Stall radius $r_{\mathrm{stall}}$ (V)")
ax.set_ylabel("Absolute error (V)")
ax.set_title(r"(b) Stall radius predictor", loc="left", fontsize=9)
ax.set_yscale("log")
ax.set_xlim(-0.05, 3.5)
ax.set_ylim(1e-4, 30)
ax.legend(fontsize=7, loc="lower right")

# (c) Minkowski vs stall-radius precision — bar chart
ax = axes[2]
b_rows = th[th["failure_mode"] == "B_compound"]
n_bc = len(b_rows)
n_mink  = int(b_rows["minkowski_holds"].sum())
n_stall = int(b_rows["r_stall_exceeds_tol"].sum())

bars = ax.bar(["Minkowski\n(Thm 1c)", "Stall radius\n(Thm 1d)"],
              [n_mink, n_stall],
              color=["#FF9800", "#F44336"], width=0.45, edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, [n_mink, n_stall]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v}/{n_bc}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel(r"$B_{\mathrm{compound}}$ circuits predicted")
ax.set_title(f"(c) Theorem 1 precision\n($n_{{B_{{\\mathrm{{compound}}}}}}={n_bc}$)", loc="left", fontsize=9)
ax.set_ylim(0, n_bc * 1.25)
ax.axhline(n_bc, color="gray", lw=0.8, ls="--", label=f"All {n_bc}")
ax.legend(fontsize=7)

fig.tight_layout(w_pad=2.5)
save(fig, "fig2_theorem1")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Timing Sweep (T1 and T2 build + inference vs N)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: timing sweep")

ts_ok2 = ts_ok.dropna(subset=["N", "T1_build_s", "T1_infer_s", "T2_build_s", "T2_infer_s"])
ts_ok2 = ts_ok2[ts_ok2["N"] > 0]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# (a) T1 (LU Transformer) timing vs n_free
ax = axes[0]
nf = ts_ok2["n_free"]
ax.scatter(nf, ts_ok2["T1_build_s"], c="#1565C0", alpha=0.55, s=14, label="Build", edgecolors="none")
ax.scatter(nf, ts_ok2["T1_infer_s"], c="#42A5F5", alpha=0.55, s=14, label="Infer", edgecolors="none")

# Quadratic fit for build
valid = ts_ok2.dropna(subset=["T1_build_s"])
nf_v  = valid["n_free"].values
b_v   = valid["T1_build_s"].values
i_v   = valid["T1_infer_s"].values
xs = np.linspace(nf_v.min(), nf_v.max(), 200)
try:
    pb = np.polyfit(np.log(nf_v + 1), np.log(b_v + 1e-9), 1)
    ax.plot(xs, np.exp(np.polyval(pb, np.log(xs + 1))), c="#0D47A1", lw=1.4,
            ls="--", label=rf"$O(n^{{{pb[0]:.1f}}})$ fit")
except Exception:
    pass
ax.set_xlabel(r"$n_{\mathrm{free}}$ (free nodes)")
ax.set_ylabel("Time (s)")
ax.set_title("(a) T1 (LU transformer) timing", loc="left", fontsize=9)
ax.legend(fontsize=7)
ax.set_xscale("log")
ax.set_yscale("log")

# (b) T2 (Readout transformer) timing vs N
ax = axes[1]
N_v = ts_ok2["N"].values
t2b = ts_ok2["T2_build_s"].values
t2i = ts_ok2["T2_infer_s"].values * 1000  # ms
ax.scatter(N_v, t2b,    c="#2E7D32", alpha=0.55, s=14, label="Build (s)",  edgecolors="none")
ax2 = ax.twinx()
ax2.scatter(N_v, t2i, c="#81C784", alpha=0.55, s=14, label="Infer (ms)", edgecolors="none")
# Linear fit for infer
try:
    pi = np.polyfit(N_v, t2i, 1)
    xs2 = np.linspace(N_v.min(), N_v.max(), 200)
    ax2.plot(xs2, np.polyval(pi, xs2), c="#1B5E20", lw=1.4, ls="--",
             label=rf"Linear fit (slope={pi[0]:.1f} ms/node)")
except Exception:
    pass
ax.set_xlabel(r"$N$ (total nodes)")
ax.set_ylabel("Build time (s)", color="#2E7D32")
ax2.set_ylabel("Infer time (ms)", color="#1B5E20")
ax.set_title("(b) T2 (Readout transformer) timing", loc="left", fontsize=9)
# Combined legend
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=7, loc="upper left")
ax2.spines["right"].set_visible(True)

fig.tight_layout(w_pad=3)
save(fig, "fig3_timing")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Head Ablation (T1 and T2)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: head ablation")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# (a) T1 head ablation — max error per (layer, head) across all circuits
ax = axes[0]
haT1c = haT1.copy()
haT1c["key"] = haT1c["layer"].astype(str) + "_" + haT1c["head"].astype(str)

# Pivot: rows = layer, cols = head, value = max global_max_err across circuits
pivot_t1 = haT1c.groupby(["layer", "head"])["global_max_err"].max().unstack(fill_value=0)
n_layers_t1 = len(pivot_t1)
n_heads_t1  = len(pivot_t1.columns)

im = ax.imshow(pivot_t1.values, aspect="auto", cmap="Reds",
               norm=LogNorm(vmin=1e-4, vmax=max(pivot_t1.values.max(), 1e-3)),
               interpolation="nearest")
ax.set_xlabel("Head index")
ax.set_ylabel("Layer")
ax.set_yticks(range(n_layers_t1))
ax.set_yticklabels([f"L{i}" for i in pivot_t1.index])
ax.set_xticks(range(0, n_heads_t1, max(1, n_heads_t1 // 8)))
ax.set_xticklabels(range(0, n_heads_t1, max(1, n_heads_t1 // 8)), fontsize=7)
ax.set_title("(a) T1 head ablation — max error\n(log scale; dark = critical)", loc="left", fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Max inv. error", fontsize=8)

# Count how many heads are critical (error > 1e-3) vs silent (error < 1e-6)
critical = (pivot_t1.values > 1e-3).sum()
silent   = (pivot_t1.values < 1e-6).sum()
total    = pivot_t1.size
ax.text(0.98, 0.04,
        f"Critical: {critical}/{total}\nSilent: {silent}/{total}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# (b) T2 head ablation — fraction of circuits where ablating that head changes pred
ax = axes[1]
haT2c = haT2.copy()
pivot_t2 = haT2c.groupby(["layer", "head"])["pred_changed"].max().unstack(fill_value=0)
n_layers_t2 = len(pivot_t2)
n_heads_t2  = len(pivot_t2.columns)

cmap_t2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#f5f5f5", "#F44336"])
im2 = ax.imshow(pivot_t2.values.astype(float), aspect="auto", cmap=cmap_t2,
                vmin=0, vmax=1, interpolation="nearest")
ax.set_xlabel("Head index")
ax.set_ylabel("Layer")
ax.set_yticks(range(n_layers_t2))
ax.set_yticklabels([f"L{i}" for i in pivot_t2.index])
ax.set_xticks(range(0, n_heads_t2, max(1, n_heads_t2 // 8)))
ax.set_xticklabels(range(0, n_heads_t2, max(1, n_heads_t2 // 8)), fontsize=7)
ax.set_title("(b) T2 head ablation — output changed\n(red = ablation changes prediction)", loc="left", fontsize=9)
cb2 = fig.colorbar(im2, ax=ax, shrink=0.8, pad=0.02)
cb2.set_label("Pred changed (max)", fontsize=8)

crit2 = int(pivot_t2.values.sum())
total2 = pivot_t2.size
ax.text(0.98, 0.04, f"Critical: {crit2}/{total2}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

fig.tight_layout(w_pad=3)
save(fig, "fig4_head_ablation")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Compression: Quantization Brittleness & Pruning Robustness
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: compression")

quant_order  = ["float64", "int16", "int8", "int4"]
prune_order  = ["prune_10", "prune_50", "prune_90"]
all_order    = quant_order + prune_order

nice_labels = {
    "float64": "float64", "int16": "int16", "int8": "int8", "int4": "int4",
    "prune_10": "prune 10%", "prune_50": "prune 50%", "prune_90": "prune 90%",
}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# (a) T1 — max error vs compression
ax = axes[0]
for ckt, color, marker in [("CKT_0015", "#1565C0", "o"), ("CKT_0067", "#F44336", "s")]:
    sub = cmT1[cmT1["circuit"] == ckt].set_index("setting")
    ys = []
    xs_present = []
    for s in all_order:
        if s in sub.index:
            ys.append(sub.loc[s, "max_err_vs_numpy"] + 1e-15)
            xs_present.append(s)
    xi = [all_order.index(s) for s in xs_present]
    ax.semilogy(xi, ys, marker=marker, color=color, ms=5, label=ckt, lw=1.2, alpha=0.9)

ax.set_xticks(range(len(all_order)))
ax.set_xticklabels([nice_labels[s] for s in all_order], rotation=30, ha="right", fontsize=7.5)
ax.axvline(3.5, color="gray", lw=0.8, ls=":", alpha=0.7)
ax.text(1.5, ax.get_ylim()[0] * 2, "Quantization", ha="center", fontsize=7.5, color="gray")
ax.text(5, ax.get_ylim()[0] * 2, "Pruning", ha="center", fontsize=7.5, color="gray")
ax.set_ylabel("Max |error| vs numpy (log)")
ax.set_title("(a) T1: Quantization brittle,\npruning robust", loc="left", fontsize=9)
ax.legend(fontsize=7)
ax.set_ylim(bottom=1e-16)

# (b) T2 — prediction error vs compression
ax = axes[1]
for ckt, color, marker in [("CKT_0015", "#1565C0", "o"), ("CKT_0067", "#F44336", "s")]:
    sub = cmT2[cmT2["circuit"] == ckt].set_index("setting")
    ys = []
    xs_present = []
    for s in all_order:
        if s in sub.index:
            ys.append(sub.loc[s, "err_vs_truth_v"] + 1e-6)
            xs_present.append(s)
    xi = [all_order.index(s) for s in xs_present]
    ax.semilogy(xi, ys, marker=marker, color=color, ms=5, label=ckt, lw=1.2, alpha=0.9)

ax.set_xticks(range(len(all_order)))
ax.set_xticklabels([nice_labels[s] for s in all_order], rotation=30, ha="right", fontsize=7.5)
ax.axvline(3.5, color="gray", lw=0.8, ls=":", alpha=0.7)
ax.axhline(0.05, color="black", lw=0.8, ls="--", alpha=0.7, label="50 mV tol")
ax.set_ylabel("Pred error vs truth (V, log)")
ax.set_title("(b) T2: Quantization brittle,\npruning robust", loc="left", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout(w_pad=3)
save(fig, "fig5_compression")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Residual Stream Probing (T1)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: residual probing T1")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# (a) T1 residual probing — abs_err per (layer, slot) across tokens
rpT1c = rpT1.copy()
if "abs_err" in rpT1c.columns and "slot" in rpT1c.columns and "layer" in rpT1c.columns:
    slot_order = sorted(rpT1c["slot"].unique(), key=lambda x: (
        0 if x.startswith("x_new") else 1 if x.startswith("x_prev") else 2))
    pivot_rp = rpT1c.groupby(["layer", "slot"])["abs_err"].mean().unstack(fill_value=np.nan)

    ax = axes[0]
    cmap_rp = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#4CAF50", "#FFEB3B", "#F44336"])
    vals = pivot_rp.values.astype(float)
    vmax = np.nanmax(vals) if np.nanmax(vals) > 0 else 1.0
    im = ax.imshow(vals, aspect="auto", cmap=cmap_rp, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Residual slot")
    ax.set_yticks(range(len(pivot_rp.index)))
    ax.set_yticklabels([f"L{i}" for i in pivot_rp.index], fontsize=8)
    ax.set_xticks(range(len(pivot_rp.columns)))
    ax.set_xticklabels(pivot_rp.columns, rotation=45, ha="right", fontsize=7)
    ax.set_title("(a) T1 residual probing\n(green=decoded, red=not yet)", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean |err|", pad=0.02)
else:
    axes[0].text(0.5, 0.5, "Insufficient T1 probing data", transform=axes[0].transAxes,
                ha="center", va="center", fontsize=9)
    axes[0].set_title("(a) T1 residual probing", loc="left", fontsize=9)

# (b) T2 residual stream norm across layers and tokens
ax = axes[1]
rpT2c = rpT2.copy()
if "l2_norm" in rpT2c.columns and "layer" in rpT2c.columns:
    tok_order = sorted(rpT2c["token"].unique(), key=lambda t: (
        0 if t == "start" else 1 if t.startswith("init") else 2 if t.startswith("v_") else 3))
    pivot_t2rp = rpT2c.groupby(["layer", "token"])["l2_norm"].mean().unstack(fill_value=np.nan)

    cols_ordered = [c for c in tok_order if c in pivot_t2rp.columns]
    pivot_t2rp = pivot_t2rp[cols_ordered] if cols_ordered else pivot_t2rp

    vals2 = pivot_t2rp.values.astype(float)
    vmax2 = np.nanmax(vals2) if np.nanmax(vals2) > 0 else 1.0
    cmap2 = "Blues"
    im2 = ax.imshow(vals2, aspect="auto", cmap=cmap2, vmin=0, vmax=vmax2, interpolation="nearest")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Token")
    ax.set_yticks(range(len(pivot_t2rp.index)))
    ax.set_yticklabels([f"L{i}" for i in pivot_t2rp.index], fontsize=8)
    ax.set_xticks(range(len(pivot_t2rp.columns)))
    ax.set_xticklabels(cols_ordered, rotation=45, ha="right", fontsize=7)
    ax.set_title("(b) T2 residual stream ‖·‖₂\nper layer × token", loc="left", fontsize=9)
    fig.colorbar(im2, ax=ax, shrink=0.8, label="Mean ‖resid‖₂", pad=0.02)
else:
    axes[1].text(0.5, 0.5, "Insufficient T2 probing data", transform=axes[1].transAxes,
                ha="center", va="center", fontsize=9)

fig.tight_layout(w_pad=3)
save(fig, "fig6_residual_probing")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Attention Patterns (T1)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: attention patterns T1")

apT1c = apT1.copy()
circuits_ap = apT1c["circuit"].unique()[:2]  # use first 2 circuits

fig, axes = plt.subplots(len(circuits_ap), 2, figsize=(7.0, 3.2 * len(circuits_ap)),
                          squeeze=False)

for row_i, ckt in enumerate(circuits_ap):
    sub = apT1c[apT1c["circuit"] == ckt]
    layers = sorted(sub["layer"].unique())

    # Left: argmax_key_pos heatmap (query_pos × layer, averaged over heads)
    ax = axes[row_i][0]
    pivot_ap = sub.groupby(["layer", "query_pos"])["argmax_key_pos"].mean().unstack(fill_value=np.nan)
    im = ax.imshow(pivot_ap.values, aspect="auto", cmap="viridis",
                   origin="upper", interpolation="nearest")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Query position")
    ax.set_yticks(range(len(pivot_ap.index)))
    ax.set_yticklabels([f"L{i}" for i in pivot_ap.index], fontsize=8)
    ax.set_title(f"({chr(97+2*row_i)}) {ckt}: argmax attended key", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Key pos", pad=0.02)

    # Right: attention entropy per (query_pos, layer)
    ax = axes[row_i][1]
    pivot_ent = sub.groupby(["layer", "query_pos"])["attn_entropy"].mean().unstack(fill_value=np.nan)
    im2 = ax.imshow(pivot_ent.values, aspect="auto", cmap="plasma",
                    origin="upper", interpolation="nearest")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Query position")
    ax.set_yticks(range(len(pivot_ent.index)))
    ax.set_yticklabels([f"L{i}" for i in pivot_ent.index], fontsize=8)
    ax.set_title(f"({chr(98+2*row_i)}) {ckt}: attn entropy (≈0 = one-hot)", loc="left", fontsize=9)
    fig.colorbar(im2, ax=ax, shrink=0.8, label="Entropy (nats)", pad=0.02)

fig.tight_layout(h_pad=2.5, w_pad=2)
save(fig, "fig7_attention_patterns")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Model Architecture Scaling (T1 and T2 together)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: architecture scaling")

ts_ok3 = ts_ok2.dropna(subset=["T1_n_params", "T2_n_params", "n_free"])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

ax = axes[0]
ax.scatter(ts_ok3["n_free"], ts_ok3["T1_n_params"] / 1e3,
           c="#1565C0", alpha=0.55, s=14, edgecolors="none")
ax.set_xlabel(r"$n_{\mathrm{free}}$ (free nodes)")
ax.set_ylabel("T1 parameter count (×10³)")
ax.set_title("(a) T1 model size vs circuit size", loc="left", fontsize=9)
ax.set_xscale("log")
ax.set_yscale("log")

ax = axes[1]
ax.scatter(ts_ok3["N"], ts_ok3["T2_n_params"] / 1e3,
           c="#2E7D32", alpha=0.55, s=14, edgecolors="none")
ax.axhline(ts_ok3["T2_n_params"].median() / 1e3, color="gray", lw=0.9, ls="--",
           label=f"Median {ts_ok3['T2_n_params'].median()/1e3:.1f}k")
ax.set_xlabel(r"$N$ (total nodes)")
ax.set_ylabel("T2 parameter count (×10³)")
ax.set_title("(b) T2 model size vs circuit size", loc="left", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout(w_pad=3)
save(fig, "fig8_model_scaling")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Overall accuracy summary (pass rate at 50mV and 75mV)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 9: accuracy summary")

total = len(ts_ok)
p50 = (ts_ok["pass_50mv"] == "PASS").sum()
p75 = (ts_ok["pass_75mv"] == "PASS").sum()

# Break down by complexity
complexities = ["Basic", "Intermediate", "Hard"]
n_by_c = {c: len(ts_ok[ts_ok["complexity"] == c]) for c in complexities}
p50_c  = {c: (ts_ok[ts_ok["complexity"] == c]["pass_50mv"] == "PASS").sum() for c in complexities}
p75_c  = {c: (ts_ok[ts_ok["complexity"] == c]["pass_75mv"] == "PASS").sum() for c in complexities}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# (a) overall bar
ax = axes[0]
cats = ["All (50 mV)", "All (75 mV)"]
vals = [p50 / total * 100, p75 / total * 100]
bars = ax.bar(cats, vals, color=["#1565C0", "#0D47A1"], width=0.4)
for bar, v, n in zip(bars, [p50, p75], [total, total]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{n - (total - round(v*total/100))}/{total}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_ylabel("Pass rate (%)")
ax.set_ylim(0, 108)
ax.set_title(f"(a) Overall pass rate\n($n={total}$ circuits)", loc="left", fontsize=9)
# annotate with actual counts
ax.text(0, p50/total*100 + 2, f"{p50}/{total}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.text(1, p75/total*100 + 2, f"{p75}/{total}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# (b) by complexity
ax = axes[1]
x = np.arange(len(complexities))
w = 0.32
b1 = ax.bar(x - w/2, [p50_c[c] / n_by_c[c] * 100 for c in complexities], w,
            color="#42A5F5", label="50 mV", edgecolor="white")
b2 = ax.bar(x + w/2, [p75_c[c] / n_by_c[c] * 100 for c in complexities], w,
            color="#1565C0", label="75 mV", edgecolor="white")
for bars_set, vals_d, cnt_d in [(b1, p50_c, n_by_c), (b2, p75_c, n_by_c)]:
    for bar, c in zip(bars_set, complexities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{vals_d[c]}/{cnt_d[c]}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(complexities)
ax.set_ylabel("Pass rate (%)")
ax.set_ylim(0, 115)
ax.set_title("(b) Pass rate by complexity", loc="left", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout(w_pad=3)
save(fig, "fig9_accuracy")

print(f"\nAll figures saved to {FIGS}/")
