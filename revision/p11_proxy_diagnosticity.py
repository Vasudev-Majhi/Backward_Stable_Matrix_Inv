"""P1.1-P1.4 / Review sec.20 (Framing 2), sec.22, sec.24 -- Proxy diagnosticity grid.

THE MAIN RESULT OF THE REFRAMED PAPER.

CONCERN / HYPOTHESIS H2b (review sec.22): "Behavioural proxies (output
finiteness, softmax one-hotness, attention entropy, prediction invariance under
pruning, head-ablation criticality) have low diagnosticity for program
correctness -- they remain 'green' in a substantial fraction of configurations
where the selector hit rate or eta has degraded catastrophically."

DESIGN. Independent variables:
    selector scale K   {1e10 (production), 1e8, 1e6, 1e4}   <- the P0 ablation
                        the review asks for: "is 1e10 necessary?"
    execution dtype    {f64, f32, bf16, f16}
    quantization       {none, per_tensor_int8, per_channel_int8}   <- P1.3
    pruning            {0, 10, 50, 90 %} of |w|, and the same fractions
                       computed over NONZERO coefficients only      <- sec.6
Dependent variables:
    GROUND TRUTH  selector hit rate vs the compiler-labelled float64 production
                  trace; per-column eta; relative forward error.
    PROXIES       fraction of exactly-one-hot softmax rows; fraction of finite
                  outputs; TRUE Shannon entropy (no 1e-30 regulariser); whether
                  the 101-point-grid prediction is unchanged from float64.

ANALYSIS. For each proxy we report the FALSE REASSURANCE RATE
    P(proxy reads green | ground truth is broken)
with cluster-bootstrap confidence intervals over matrices -- the statistical
unit is the (matrix, configuration) cell, NOT the pooled lookup event, which is
the pseudoreplication the review flags in sec.9.2.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from craft_harness import (  # noqa: E402
    CompiledSolver, build_model_for, circuit_system, eta_rigal_gaches,
    host_lu_solve, load_circuits, K_LEVELS, SCALE, V_STEP,
)

RES = os.path.join(HERE, "results")
DTYPES = {"f64": torch.float64, "f32": torch.float32,
          "bf16": torch.bfloat16, "f16": torch.float16}


def quantize_v(v):
    k = int(round(v * SCALE / V_STEP))
    return max(0, min(K_LEVELS - 1, k)) * V_STEP / SCALE


def build_with_K(cid, K, out_dir):
    """Build a model with the selector scale K patched at compile time."""
    import transformer_vm.model.weights as W
    old = W.HARD_K
    W.HARD_K = float(K)
    try:
        tag = f"K{K:.0e}".replace("+", "")
        d = os.path.join(out_dir, tag)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"model_{cid}_lu_direct.bin")
        if not (os.path.exists(path) and os.path.exists(path + ".slots.json")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                from build_lu_direct import build_for_circuit
                build_for_circuit(cid, out_path=path)
        return path
    finally:
        W.HARD_K = old


def evaluate(S, A, b_scaled, ref_sel, ref_pred):
    r = S.solve(b_scaled, record=True)
    c = r["cache"]
    x = np.asarray(r["x"], dtype=np.float64)
    finite_x = bool(np.all(np.isfinite(x)))
    eta = eta_rigal_gaches(A, x, b_scaled) if finite_x else float("inf")
    xh = host_lu_solve(A, b_scaled)
    den = max(np.max(np.abs(xh)), 1e-300)
    fwd = (float(np.max(np.abs(x - xh)) / den) if finite_x else float("inf"))
    m = min(len(c.sel), len(ref_sel))
    hit = float(np.mean([c.sel[i] == ref_sel[i] for i in range(m)])) if m else float("nan")
    return dict(
        selector_hit=hit,
        eta=eta,
        fwd_rel_err=fwd,
        pred_v=r["pred_v"],
        pred_unchanged=bool(abs(r["pred_v"] - ref_pred) < 1e-12),
        frac_onehot=float(np.mean(c.onehot)) if c.onehot else float("nan"),
        frac_finite=float(np.mean(c.finite)) if c.finite else float("nan"),
        mean_entropy=float(np.mean(c.entropy)) if c.entropy else float("nan"),
        max_entropy=float(np.max(c.entropy)) if c.entropy else float("nan"),
        frac_tied=float(np.mean(c.tied)) if c.tied else float("nan"),
        n_lookups=len(c.sel),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=12)
    ap.add_argument("--min-free", type=int, default=3)
    ap.add_argument("--k-scales", type=float, nargs="+",
                    default=[1e10, 1e8, 1e6, 1e4])
    args = ap.parse_args()

    circuits = load_circuits()
    picked = []
    for cid, c in circuits.items():
        A, b, free, fixed, pc = circuit_system(c["Netlist"])
        if len(free) >= args.min_free:
            picked.append((cid, c, A, b, free))
    picked.sort(key=lambda t: len(t[4]))
    if len(picked) > args.n_circuits:
        idx = np.linspace(0, len(picked) - 1, args.n_circuits).astype(int)
        picked = [picked[i] for i in idx]

    rows = []
    for ci, (cid, c, A, b, free) in enumerate(picked):
        bs = b * 1e4
        for K in args.k_scales:
            try:
                mp_ = build_with_K(cid, K, os.path.join(HERE, "models_K"))
            except Exception as e:
                print(f"[skip build] {cid} K={K:g}: {e}")
                continue
            # production float64 reference trace for THIS K
            try:
                ref = CompiledSolver(mp_, c["Netlist"], dtype=torch.float64).solve(bs)
            except Exception as e:
                print(f"[skip ref] {cid} K={K:g}: {e}")
                continue
            ref_sel, ref_pred = list(ref["cache"].sel), ref["pred_v"]

            configs = []
            for dt in DTYPES:
                configs.append(dict(dtype=dt, quant=None, prune=0.0, prune_nz=False))
            for q in ("per_tensor_int8", "per_channel_int8"):
                configs.append(dict(dtype="f64", quant=q, prune=0.0, prune_nz=False))
            for pf in (0.10, 0.50, 0.90):
                configs.append(dict(dtype="f64", quant=None, prune=pf, prune_nz=False))
                configs.append(dict(dtype="f64", quant=None, prune=pf, prune_nz=True))

            for cfg in configs:
                try:
                    S = CompiledSolver(
                        mp_, c["Netlist"], dtype=DTYPES[cfg["dtype"]],
                        prune_frac=cfg["prune"], prune_nonzero_only=cfg["prune_nz"],
                        quantize=cfg["quant"],
                    )
                    m = evaluate(S, A, bs, ref_sel, ref_pred)
                except Exception as e:
                    m = dict(selector_hit=float("nan"), eta=float("inf"),
                             fwd_rel_err=float("inf"), pred_v=float("nan"),
                             pred_unchanged=False, frac_onehot=float("nan"),
                             frac_finite=0.0, mean_entropy=float("nan"),
                             max_entropy=float("nan"), frac_tied=float("nan"),
                             n_lookups=0, error=f"{type(e).__name__}: {e}")
                rows.append(dict(
                    circuit=cid, n_free=len(free), K=K, **cfg, **m))
        print(f"[{ci+1}/{len(picked)}] {cid} n={len(free)} done "
              f"({len([r for r in rows if r['circuit']==cid])} cells)", flush=True)

    os.makedirs(RES, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RES, "p11_proxy_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- diagnosticity analysis ---------------------------------------
    u = np.finfo(np.float64).eps / 2
    ETA_BROKEN = 1e-10        # eta far above any correct binary64 solve
    HIT_BROKEN = 1.0          # any selector miss at all is a broken program

    def broken(r):
        return (not np.isfinite(r["eta"])) or r["eta"] > ETA_BROKEN or \
               (np.isfinite(r["selector_hit"]) and r["selector_hit"] < HIT_BROKEN)

    proxies = {
        "outputs_all_finite": lambda r: r["frac_finite"] >= 1.0,
        "softmax_exactly_onehot": lambda r: r["frac_onehot"] >= 1.0,
        "attention_entropy_near_zero": lambda r: (np.isfinite(r["max_entropy"])
                                                  and r["max_entropy"] < 1e-6),
        "prediction_unchanged_on_grid": lambda r: bool(r["pred_unchanged"]),
    }
    res = {"n_cells": len(rows),
           "n_matrices": len({r["circuit"] for r in rows}),
           "eta_broken_threshold": ETA_BROKEN, "u": u,
           "n_broken": int(sum(broken(r) for r in rows)),
           "proxies": {}}
    bro = [r for r in rows if broken(r)]
    rng = np.random.default_rng(0)
    mats = sorted({r["circuit"] for r in rows})
    for pname, pf in proxies.items():
        green_given_broken = [bool(pf(r)) for r in bro]
        rate = float(np.mean(green_given_broken)) if green_given_broken else float("nan")
        # cluster bootstrap over matrices
        boots = []
        for _ in range(2000):
            sel = rng.choice(len(mats), size=len(mats), replace=True)
            pool = [r for i in sel for r in bro if r["circuit"] == mats[i]]
            if pool:
                boots.append(np.mean([bool(pf(r)) for r in pool]))
        ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
              if boots else [float("nan")] * 2)
        res["proxies"][pname] = {
            "false_reassurance_rate": rate,
            "cluster_bootstrap_95CI": ci,
            "n_broken_cells": len(bro),
            "n_green_while_broken": int(sum(green_given_broken)),
        }
    # worst cell: proxy-green but maximally broken
    green_all = [r for r in bro if all(pf(r) for pf in proxies.values())]
    res["all_four_proxies_green_while_broken"] = {
        "n": len(green_all),
        "examples": sorted(
            [{k: r[k] for k in ("circuit", "K", "dtype", "quant", "prune",
                                "prune_nz", "selector_hit", "eta")}
             for r in green_all],
            key=lambda d: -d["eta"])[:10],
    }
    # K ablation: is 1e10 necessary in float64?
    res["K_ablation_float64"] = {}
    for K in args.k_scales:
        sub = [r for r in rows if r["K"] == K and r["dtype"] == "f64"
               and r["quant"] is None and r["prune"] == 0.0]
        if sub:
            res["K_ablation_float64"][f"{K:.0e}"] = {
                "n": len(sub),
                "max_eta": max(r["eta"] for r in sub),
                "min_selector_hit": min(r["selector_hit"] for r in sub),
            }
    with open(os.path.join(RES, "p11_summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
