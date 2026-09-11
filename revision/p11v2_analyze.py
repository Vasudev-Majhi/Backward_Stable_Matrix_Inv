"""Analysis for the v2 grid: discriminability (W5), ablation 2x2 (W6), figure.

W5 said a bare false-reassurance rate P(green | broken) is the wrong headline:
a proxy that is green almost everywhere attains a high value trivially. We
therefore report, for each proxy:

  P(green | correct)      does the proxy stay green when the program is right?
  P(green | broken)       the false-reassurance rate (kept, but not alone)
  LR_green                P(green|correct) / P(green|broken); 1.0 = uninformative
  AUC                     rank discriminability of the proxy's CONTINUOUS value
                          against compiler-labelled correctness; 0.5 = chance
  Youden J                sensitivity + specificity - 1 at the natural threshold

All with a cluster bootstrap over matrices (the resampling unit is the matrix,
not the cell), and the grid now has no K replication, so cells are independent.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def fnum(x, k, default=float("nan")):
    try:
        return float(x[k])
    except (TypeError, ValueError, KeyError):
        return default


def auc(pos_scores, neg_scores):
    """Probability a random 'correct' run scores above a random 'broken' one.
    Ties counted as 0.5. Returns nan if either group is empty."""
    p = np.asarray([v for v in pos_scores if np.isfinite(v)], dtype=float)
    n = np.asarray([v for v in neg_scores if np.isfinite(v)], dtype=float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rp = ranks[:p.size].sum()
    return float((rp - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def main():
    with open(os.path.join(RES, "p11v2_grid.csv")) as f:
        rows = list(csv.DictReader(f))

    # ---- ground truth: the certified-exact compiler oracle ---------------
    def broken(x):
        h = fnum(x, "selector_hit")
        e = fnum(x, "eta", float("inf"))
        return (np.isfinite(h) and h < 1.0) or (not np.isfinite(e))

    for x in rows:
        x["_broken"] = broken(x)

    # proxies: (name, continuous value, green predicate) -- higher = healthier
    PROX = {
        "outputs all finite": (lambda x: fnum(x, "frac_finite"),
                               lambda x: fnum(x, "frac_finite") >= 1.0),
        "softmax exactly one-hot": (lambda x: fnum(x, "frac_onehot"),
                                    lambda x: fnum(x, "frac_onehot") >= 1.0),
        "attention entropy ~ 0": (lambda x: -fnum(x, "mean_entropy"),
                                  lambda x: fnum(x, "mean_entropy") < 1e-9),
        "prediction unchanged": (lambda x: 1.0 if x["pred_unchanged"] == "True" else 0.0,
                                 lambda x: x["pred_unchanged"] == "True"),
    }

    mats = sorted({x["circuit"] for x in rows})
    hea = [x for x in rows if not x["_broken"]]
    bro = [x for x in rows if x["_broken"]]

    def stats(sample):
        h = [x for x in sample if not x["_broken"]]
        b = [x for x in sample if x["_broken"]]
        out = {}
        for name, (val, grn) in PROX.items():
            pg_h = np.mean([grn(x) for x in h]) if h else np.nan
            pg_b = np.mean([grn(x) for x in b]) if b else np.nan
            out[name] = dict(
                p_green_correct=float(pg_h), p_green_broken=float(pg_b),
                lr_green=float(pg_h / pg_b) if pg_b > 0 else float("inf"),
                auc=auc([val(x) for x in h], [val(x) for x in b]),
                youden=float(pg_h - pg_b),
            )
        return out

    point = stats(rows)
    rng = np.random.default_rng(0)
    boots = {k: {m: [] for m in point[k]} for k in point}
    for _ in range(4000):
        sel = rng.choice(len(mats), len(mats), replace=True)
        pool = [x for i in sel for x in rows if x["circuit"] == mats[i]]
        s = stats(pool)
        for k in s:
            for m in s[k]:
                v = s[k][m]
                if np.isfinite(v):
                    boots[k][m].append(v)

    res = {
        "n_cells": len(rows), "n_matrices": len(mats),
        "n_broken": len(bro), "n_correct": len(hea),
        "grid_has_no_K_replication": True,
        "ground_truth": "certified-exact selector oracle on compiler-labelled lookup heads",
        "proxies": {},
    }
    for k in point:
        res["proxies"][k] = dict(point[k])
        for m, arr in boots[k].items():
            if arr:
                res["proxies"][k][m + "_95CI"] = [float(np.percentile(arr, 2.5)),
                                                  float(np.percentile(arr, 97.5))]
    allg = [x for x in bro if all(g(x) for _, g in PROX.values())]
    res["all_proxies_green_while_broken"] = {
        "n": len(allg), "rate": len(allg) / len(bro) if bro else float("nan")}

    # ---- per-configuration table ---------------------------------------
    cfg = []
    for lab in sorted({x["label"] for x in rows}):
        s = [x for x in rows if x["label"] == lab]
        cfg.append(dict(
            label=lab, n=len(s),
            frac_broken=float(np.mean([x["_broken"] for x in s])),
            median_hit=float(np.median([fnum(x, "selector_hit") for x in s])),
            median_eta=float(np.median([fnum(x, "eta", np.inf) for x in s])),
            **{k: float(np.mean([g(x) for x in s])) for k, (_, g) in PROX.items()},
        ))
    res["by_configuration"] = cfg

    # ---- oracle certification -------------------------------------------
    op = os.path.join(RES, "p11v2_oracle.csv")
    if os.path.exists(op):
        with open(op) as f:
            orows = list(csv.DictReader(f))
        res["oracle"] = {
            "n_matrices": len(orows),
            "mean_frac_certified_lookup": float(np.mean([fnum(x, "frac_certified_lookup") for x in orows])),
            "min_frac_certified_lookup": float(np.min([fnum(x, "frac_certified_lookup") for x in orows])),
            "total_lookup_decisions": int(sum(fnum(x, "n_lookup_scored") for x in orows)),
            "total_exact_fallbacks": int(sum(fnum(x, "n_exact_fallback") for x in orows)),
        }

    # ---- W6: head-ablation 2x2 against compiler labels -------------------
    ap = os.path.join(RES, "p11v2_ablation.csv")
    if os.path.exists(ap):
        with open(ap) as f:
            arows = list(csv.DictReader(f))
        tab = {}
        for lab in ("lookup", "passthrough"):
            sub = [x for x in arows if x["compiler_label"] == lab]
            if not sub:
                continue
            tab[lab] = {
                "n_heads": len(sub),
                "prediction_changing": int(sum(x["pred_changed"] == "True" for x in sub)),
                "program_breaking": int(sum(x["breaks_program"] == "True" for x in sub)),
                "frac_prediction_changing": float(np.mean([x["pred_changed"] == "True" for x in sub])),
                "frac_program_breaking": float(np.mean([x["breaks_program"] == "True" for x in sub])),
            }
        crit = [x for x in arows if x["pred_changed"] == "True"]
        lk = [x for x in arows if x["compiler_label"] == "lookup"]
        res["head_ablation"] = {
            "by_compiler_label": tab,
            "n_heads_total": len(arows),
            "n_circuits": len({x["circuit"] for x in arows}),
            "precision_of_ablation_criticality": (
                float(np.mean([x["compiler_label"] == "lookup" for x in crit])) if crit else float("nan")),
            "recall_of_ablation_criticality": (
                float(np.mean([x["pred_changed"] == "True" for x in lk])) if lk else float("nan")),
            "note": ("precision = fraction of prediction-changing heads the compiler "
                     "labels as lookup; recall = fraction of compiler-labelled lookup "
                     "heads whose ablation changes the prediction"),
        }

    with open(os.path.join(RES, "p11v2_diagnosticity.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg.sort(key=lambda d: (d["frac_broken"], -d["median_hit"]))
    xs = np.arange(len(cfg))
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    eta = np.array([max(d["median_eta"], 1e-18) for d in cfg])
    ax.set_yscale("log")
    bad = [i for i, d in enumerate(cfg) if d["frac_broken"] > 0]
    if bad:
        ax.axvspan(min(bad) - 0.5, len(cfg) - 0.5, color="#c1121f", alpha=0.06, zorder=0)
        ax.text((min(bad) + len(cfg) - 1) / 2, 3e2, "selector fetches the WRONG value",
                ha="center", va="top", fontsize=12, color="#c1121f",
                style="italic", weight="bold")
        ax.text((min(bad) - 1) / 2, 3e2, "program correct", ha="center", va="top",
                fontsize=12, color="#2a7221", style="italic", weight="bold")
    ax.plot(xs, eta, "o-", color="#c1121f", lw=2.8, ms=8, zorder=6,
            label=r"GROUND TRUTH: median per-column $\eta$")
    u = np.finfo(np.float64).eps / 2
    ax.axhline(u, color="#c1121f", ls=":", lw=1.4)
    ax.text(-0.45, u * 1.7, r"$u$", color="#c1121f", fontsize=10)
    ax.set_ylabel(r"backward error $\eta$ (log)", color="#c1121f", fontsize=12)
    ax.tick_params(axis='y', labelcolor='#c1121f', labelsize=10)
    ax.set_ylim(1e-18, 1e3)
    ax2 = ax.twinx()
    styles = [("outputs all finite", "#0353a4", "s", 0.0),
              ("softmax exactly one-hot", "#009e73", "^", 0.012),
              ("attention entropy ~ 0", "#7209b7", "v", -0.012),
              ("prediction unchanged", "#e08e00", "D", 0.024)]
    for name, col, mk, dy in styles:
        ax2.plot(xs, [d[name] + dy for d in cfg], marker=mk, ls="--", color=col,
                 lw=1.9, ms=6.5, alpha=.92, zorder=4, label=f"proxy: {name}")
    ax2.set_ylabel("fraction reading GREEN", fontsize=12)
    ax2.set_ylim(-0.08, 1.13)
    ax2.tick_params(axis="y", labelsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([d["label"] for d in cfg], rotation=30, ha="right", fontsize=11)
    ax.set_xlim(-0.6, len(cfg) - 0.4)
    ax.set_xlabel("execution configuration, ordered by compiler ground truth", fontsize=12, labelpad=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.58),
              ncol=3, fontsize=10.5, framealpha=.97, columnspacing=1.1,
              handlelength=1.8)
    ax.grid(alpha=.22, axis="y")
    fig.subplots_adjust(bottom=0.42, top=0.95, left=0.085, right=0.915)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig1_proxy_dissociation.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "fig1_proxy_dissociation.pdf"))
    print(json.dumps({k: res[k] for k in
                      ("n_cells", "n_matrices", "n_broken", "proxies",
                       "all_proxies_green_while_broken", "oracle", "head_ablation")
                      if k in res}, indent=2, default=str))


if __name__ == "__main__":
    main()
