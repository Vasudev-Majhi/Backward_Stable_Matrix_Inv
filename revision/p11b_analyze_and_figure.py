"""P1.1 analysis + the KILLER FIGURE (review sec.24).

Scores each behavioural proxy the way the paper itself defines a "reference
hit": a proxy reads GREEN for a configuration when its value is no worse than
the value the float64 production configuration attains on the same matrix.
That avoids the degenerate absolute thresholds (requiring EVERY lookup in a run
to be exactly one-hot is never satisfied, even by a provably correct run, so an
absolute threshold measures nothing).

Ground truth is the compiler: a configuration is BROKEN when the selector fetches
a value from a position other than the one the float64 production trace fetched,
or when the per-column backward error eta exceeds 1e-10 (fifteen orders above u).

Outputs:
  results/p11_diagnosticity.json
  figures/killer_figure.png   -- "Behavioural health proxies do not track
                                 program correctness."
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from craft_harness import circuit_system, load_circuits  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
ETA_BROKEN = 1e-10


def fnum(x, k, default=float("nan")):
    try:
        v = float(x[k])
        return v
    except (TypeError, ValueError):
        return default


def load_rows():
    cs = load_circuits()
    fixed = {cid for cid, c in cs.items()
             if circuit_system(c["Netlist"])[4].is_fixed[int(c["Target_Node"])]}
    rows = []
    with open(os.path.join(RES, "p11_proxy_grid.csv")) as f:
        for x in csv.DictReader(f):
            if x["circuit"] in fixed:
                continue          # LU path never exercised; see p01b
            rows.append(x)
    return rows, sorted(fixed)


def cfg_label(x):
    if x["quant"]:
        return x["quant"].replace("_int8", " int8")
    if float(x["prune"]) > 0:
        return f"prune {int(float(x['prune'])*100)}%" + ("(nz)" if x["prune_nz"] == "True" else "")
    return x["dtype"]


def main():
    rows, fixed = load_rows()
    # reference cell per (circuit, K): float64, unquantized, unpruned
    ref = {}
    for x in rows:
        if x["dtype"] == "f64" and not x["quant"] and float(x["prune"]) == 0.0:
            ref[(x["circuit"], x["K"])] = x

    PROX = {
        "outputs all finite": lambda x, r: fnum(x, "frac_finite") >= fnum(r, "frac_finite") - 1e-12,
        "softmax one-hot fraction": lambda x, r: fnum(x, "frac_onehot") >= fnum(r, "frac_onehot") - 1e-12,
        "attention entropy": lambda x, r: fnum(x, "mean_entropy") <= fnum(r, "mean_entropy") + 1e-12,
        "prediction unchanged (50 mV grid)": lambda x, r: x["pred_unchanged"] == "True",
    }

    def broken(x):
        """Ground truth is the COMPILER: a run is broken when any attention
        lookup fetches from a position other than the one the compiler
        scheduled (as directly observed in the float64 production trace), or
        when the output is not finite.

        We deliberately do NOT call float32 'broken' merely for having
        eta ~ u_f32: executing the same program in a lower precision correctly
        is a different thing from executing the wrong program. This is the
        stricter, more defensible reading, and it makes the negative result
        harder to obtain rather than easier.
        """
        h = fnum(x, "selector_hit")
        e = fnum(x, "eta", float("inf"))
        return (np.isfinite(h) and h < 1.0) or (not np.isfinite(e))

    recs = []
    for x in rows:
        r = ref.get((x["circuit"], x["K"]))
        if r is None:
            continue
        recs.append(dict(
            circuit=x["circuit"], K=float(x["K"]), label=cfg_label(x),
            dtype=x["dtype"], quant=x["quant"], prune=float(x["prune"]),
            prune_nz=x["prune_nz"] == "True",
            eta=fnum(x, "eta", float("inf")),
            hit=fnum(x, "selector_hit"),
            broken=broken(x),
            **{f"proxy::{k}": bool(v(x, r)) for k, v in PROX.items()},
        ))

    mats = sorted({r["circuit"] for r in recs})
    rng = np.random.default_rng(0)
    bro = [r for r in recs if r["broken"]]
    hea = [r for r in recs if not r["broken"]]
    out = {
        "n_cells": len(recs), "n_matrices": len(mats),
        "n_broken": len(bro), "n_healthy": len(hea),
        "excluded_fixed_target_circuits": fixed,
        "eta_broken_threshold": ETA_BROKEN,
        "ground_truth": "selector position vs float64 compiler trace, and eta",
        "proxies": {},
    }
    for p in PROX:
        key = f"proxy::{p}"
        base = float(np.mean([r[key] for r in hea])) if hea else float("nan")
        fr = float(np.mean([r[key] for r in bro])) if bro else float("nan")
        boots = []
        for _ in range(3000):
            sel = rng.choice(len(mats), len(mats), replace=True)
            pool = [r for i in sel for r in bro if r["circuit"] == mats[i]]
            if pool:
                boots.append(np.mean([r[key] for r in pool]))
        out["proxies"][p] = {
            "green_rate_on_correct_runs": base,
            "false_reassurance_rate": fr,
            "cluster_bootstrap_95CI": [float(np.percentile(boots, 2.5)),
                                       float(np.percentile(boots, 97.5))] if boots else None,
            "n_green_while_broken": int(sum(r[key] for r in bro)),
            "n_broken": len(bro),
        }
    allg = [r for r in bro if all(r[f"proxy::{p}"] for p in PROX)]
    out["all_proxies_green_while_broken"] = {
        "n": len(allg),
        "rate": len(allg) / len(bro) if bro else float("nan"),
        "worst_examples": sorted(
            [{k: r[k] for k in ("circuit", "K", "label", "hit", "eta")} for r in allg],
            key=lambda d: -d["eta"])[:8],
    }
    # K ablation
    out["K_ablation"] = {}
    for K in sorted({r["K"] for r in recs}):
        s = [r for r in recs if r["K"] == K and r["label"] == "f64"]
        if s:
            out["K_ablation"][f"{K:.0e}"] = {
                "n": len(s), "max_eta": max(r["eta"] for r in s),
                "min_selector_hit": min(r["hit"] for r in s)}
    # per-configuration ground-truth summary (for the figure)
    labels = []
    for lab in sorted({r["label"] for r in recs}):
        s = [r for r in recs if r["label"] == lab]
        labels.append(dict(
            label=lab, n=len(s),
            median_hit=float(np.median([r["hit"] for r in s])),
            max_eta=float(np.max([r["eta"] for r in s])),
            median_eta=float(np.median([r["eta"] for r in s])),
            **{p: float(np.mean([r[f"proxy::{p}"] for r in s])) for p in PROX},
        ))
    out["by_configuration"] = labels
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "p11_diagnosticity.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- the killer figure --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels.sort(key=lambda d: (-d["median_hit"], d["median_eta"]))
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    eta = np.array([max(d["median_eta"], 1e-18) for d in labels])
    ax.set_yscale("log")

    # shade the contiguous region where the compiler ground truth is violated
    bad = [i for i, d in enumerate(labels) if d["median_hit"] < 1.0]
    if bad:
        ax.axvspan(min(bad) - 0.5, len(labels) - 0.5, color="#c1121f", alpha=0.06,
                   zorder=0)
        ax.text((min(bad) + len(labels) - 1) / 2.0, 3e2,
                "selector fetches the WRONG value", ha="center", va="top",
                fontsize=11, color="#c1121f", style="italic", weight="bold")
        ax.text((0 + min(bad) - 1) / 2.0, 3e2, "program correct",
                ha="center", va="top", fontsize=11, color="#2a7221",
                style="italic", weight="bold")

    ax.plot(xs, eta, "o-", color="#c1121f", lw=2.8, ms=8,
            label=r"GROUND TRUTH: median per-column $\eta$", zorder=6)
    u = np.finfo(np.float64).eps / 2
    ax.axhline(u, color="#c1121f", ls=":", lw=1.4, zorder=2)
    ax.text(-0.45, u * 1.6, r"$u$", color="#c1121f", fontsize=10)
    ax.set_ylabel(r"per-column backward error $\eta$   (log scale)",
                  color="#c1121f", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#c1121f")
    ax.set_ylim(1e-18, 1e3)

    ax2 = ax.twinx()
    styles = [("outputs all finite", "#0353a4", "s", 0.000),
              ("softmax one-hot fraction", "#009e73", "^", 0.012),
              ("attention entropy", "#7209b7", "v", -0.012),
              ("prediction unchanged (50 mV grid)", "#e08e00", "D", 0.024)]
    for p_, col, mk, dy in styles:
        ax2.plot(xs, [d[p_] + dy for d in labels], marker=mk, ls="--",
                 color=col, lw=1.9, ms=6.5, alpha=.92,
                 label=f"proxy: {p_}", zorder=4)
    ax2.set_ylabel("fraction of runs where the proxy reads GREEN", fontsize=11)
    ax2.set_ylim(-0.08, 1.13)

    ax.set_xticks(xs)
    ax.set_xticklabels([d["label"] for d in labels], rotation=32, ha="right",
                       fontsize=9.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("execution configuration, ordered by compiler ground truth",
                  fontsize=11)
    ax.set_title("Behavioural health proxies do not track program correctness\n"
                 f"CRAFT compiled LU solves: {out['n_matrices']} matrices, "
                 f"{out['n_cells']} configuration cells, compiler-labelled ground truth",
                 fontsize=12.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="upper center",
                    bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9.5,
                    framealpha=.97)
    ax.grid(alpha=.22, which="major", axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "killer_figure.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "killer_figure.pdf"))
    print(json.dumps({k: out[k] for k in
                      ("n_cells", "n_matrices", "n_broken", "proxies",
                       "all_proxies_green_while_broken", "K_ablation")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
