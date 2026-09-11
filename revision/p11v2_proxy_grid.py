"""Main experiment, v2 -- rebuilt to answer reviewer weaknesses W2-W6, W8.

W2  The v1 grid crossed K in {1e10,1e8,1e6,1e4} with everything else and reported
    480 "cells". K provably has no effect (it cancels from the margin condition),
    and the replicates agreed on the correctness label 100% of the time, so the
    effective sample was 120, not 480, and the cluster-bootstrap intervals were
    correspondingly too tight. Here the main grid has NO K axis; the K ablation is
    a separate, smaller table reported on its own.

W3  Proxies are scored on compiler-labelled LOOKUP heads only. Pooling in
    passthrough heads (which legitimately attend broadly) capped `frac_onehot` at
    0.22 even on provably correct runs, making the proxy meaningless.

W4  Ground truth is the certified-exact selector oracle (craft_groundtruth.py),
    a property of the compiled program, not of a float64 run.

W5  We report discriminability -- P(green | correct), P(green | broken),
    likelihood ratios and AUC -- rather than a bare false-reassurance rate, which
    a near-constant proxy attains trivially.

W6  Head-ablation criticality is measured here (it was claimed but absent in v1)
    and cross-tabulated against the compiler's own head labels.

W8  Scaled from 10 to as many matrices as the budget allows.
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
    host_lu_solve, load_circuits,
)
from craft_groundtruth import (  # noqa: E402
    build_oracle, head_labels_for, score_against_oracle,
)

RES = os.path.join(HERE, "results")
DTYPES = {"f64": torch.float64, "f32": torch.float32,
          "bf16": torch.bfloat16, "f16": torch.float16}


def configs():
    out = [dict(dtype=d, quant=None, prune=0.0, prune_nz=False) for d in DTYPES]
    for q in ("per_tensor_int8", "per_channel_int8"):
        out.append(dict(dtype="f64", quant=q, prune=0.0, prune_nz=False))
    for pf in (0.10, 0.50, 0.90):
        out.append(dict(dtype="f64", quant=None, prune=pf, prune_nz=False))
        out.append(dict(dtype="f64", quant=None, prune=pf, prune_nz=True))
    return out


def label_of(cfg):
    if cfg["quant"]:
        return cfg["quant"].replace("_int8", " int8")
    if cfg["prune"] > 0:
        return f"prune {int(cfg['prune']*100)}%" + ("(nz)" if cfg["prune_nz"] else "")
    return cfg["dtype"]


def measure(S, A, bs, oracle, labels, ref_pred):
    r = S.solve(bs, record=True)
    c = r["cache"]
    x = np.asarray(r["x"], dtype=np.float64)
    finite_x = bool(np.all(np.isfinite(x)))
    eta = eta_rigal_gaches(A, x, bs) if finite_x else float("inf")
    xh = host_lu_solve(A, bs)
    den = max(np.max(np.abs(xh)), 1e-300)
    fwd = float(np.max(np.abs(x - xh)) / den) if finite_x else float("inf")
    hit, n_scored = score_against_oracle(c.sel, c.lh, oracle, labels, "lookup")
    keep = [i for i in range(len(c.lh))
            if labels is None or labels.get(c.lh[i]) == "lookup"]
    oh = [c.onehot[i] for i in keep]
    fin = [c.finite[i] for i in keep]
    ent = [c.entropy[i] for i in keep]
    return dict(
        selector_hit=hit, n_lookups_scored=n_scored,
        eta=eta, fwd_rel_err=fwd, pred_v=r["pred_v"],
        pred_unchanged=bool(abs(r["pred_v"] - ref_pred) < 1e-12),
        frac_onehot=float(np.mean(oh)) if oh else float("nan"),
        frac_finite=float(np.mean(fin)) if fin else float("nan"),
        mean_entropy=float(np.mean(ent)) if ent else float("nan"),
        max_entropy=float(np.max(ent)) if ent else float("nan"),
    )


def head_ablation(S, A, bs, oracle, labels, ref_pred, ref_eta):
    """Zero each head's output contribution in turn; record whether the
    prediction changes and whether the program breaks. Cross-tabulated against
    the compiler's label for that head."""
    rows = []
    n_layers = len(S.model.attn)
    n_heads = S.model.attn[0].num_heads
    d_head = S.model.attn[0].out_proj.weight.shape[1] // n_heads
    for li in range(n_layers):
        W = S.model.attn[li].out_proj.weight
        for h in range(n_heads):
            lab = labels.get((li, h), "unused") if labels else "unknown"
            sl = slice(h * d_head, (h + 1) * d_head)
            saved = W[:, sl].detach().clone()
            with torch.no_grad():
                W[:, sl] = 0.0
            try:
                m = measure(S, A, bs, oracle, labels, ref_pred)
            except Exception:
                m = dict(selector_hit=float("nan"), eta=float("inf"),
                         pred_unchanged=False)
            finally:
                with torch.no_grad():
                    W[:, sl] = saved
            broke = ((not np.isfinite(m["eta"])) or m["eta"] > 1e-10
                     or (np.isfinite(m["selector_hit"]) and m["selector_hit"] < 1.0))
            rows.append(dict(layer=li, head=h, compiler_label=lab,
                             pred_changed=not m["pred_unchanged"],
                             breaks_program=bool(broke),
                             eta=m["eta"], selector_hit=m["selector_hit"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=30)
    ap.add_argument("--min-free", type=int, default=3)
    ap.add_argument("--ablate-circuits", type=int, default=6)
    ap.add_argument("--k-ablation-circuits", type=int, default=8)
    args = ap.parse_args()

    circuits = load_circuits()
    picked = []
    for cid, c in circuits.items():
        A, b, free, fixed, pc = circuit_system(c["Netlist"])
        if pc.is_fixed[int(c["Target_Node"])]:
            continue                      # LU path never exercised (see paper 2)
        if len(free) >= args.min_free:
            picked.append((cid, c, A, b, free))
    picked.sort(key=lambda t: len(t[4]))
    if len(picked) > args.n_circuits:
        idx = np.linspace(0, len(picked) - 1, args.n_circuits).astype(int)
        picked = [picked[i] for i in idx]

    grid_rows, abl_rows, oracle_rows = [], [], []
    for ci, (cid, c, A, b, free) in enumerate(picked):
        bs = b * 1e4
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                mp_ = build_model_for(cid, os.path.join(HERE, "models_gt"))
            labels = head_labels_for(mp_)
            S0 = CompiledSolver(mp_, c["Netlist"], dtype=torch.float64)
            oracle = build_oracle(S0, bs, labels=labels)
            ref = S0.solve(bs, record=True)
            ref_pred = ref["pred_v"]
            ref_eta = eta_rigal_gaches(A, ref["x"], bs)
        except Exception as e:
            print(f"[skip] {cid}: {type(e).__name__}: {e}", flush=True)
            continue

        oracle_rows.append(dict(
            circuit=cid, n_free=len(free),
            n_lookup_heads=sum(1 for v in (labels or {}).values() if v == "lookup"),
            n_passthrough_heads=sum(1 for v in (labels or {}).values() if v == "passthrough"),
            n_lookups_total=oracle["n_lookups"],
            n_lookup_scored=oracle["n_lookup_heads_scored"],
            frac_certified_lookup=oracle["frac_certified_lookup"],
            n_exact_fallback=oracle["n_exact_fallback"],
        ))

        for cfg in configs():
            try:
                S = CompiledSolver(mp_, c["Netlist"], dtype=DTYPES[cfg["dtype"]],
                                   prune_frac=cfg["prune"],
                                   prune_nonzero_only=cfg["prune_nz"],
                                   quantize=cfg["quant"])
                m = measure(S, A, bs, oracle, labels, ref_pred)
            except Exception as e:
                m = dict(selector_hit=float("nan"), n_lookups_scored=0,
                         eta=float("inf"), fwd_rel_err=float("inf"),
                         pred_v=float("nan"), pred_unchanged=False,
                         frac_onehot=float("nan"), frac_finite=0.0,
                         mean_entropy=float("nan"), max_entropy=float("nan"),
                         error=f"{type(e).__name__}: {e}")
            grid_rows.append(dict(circuit=cid, n_free=len(free),
                                  label=label_of(cfg), **cfg, **m))

        if ci < args.ablate_circuits:
            abl_rows += [dict(circuit=cid, **r) for r in
                         head_ablation(S0, A, bs, oracle, labels, ref_pred, ref_eta)]
        print(f"[{ci+1}/{len(picked)}] {cid} n={len(free)} "
              f"certified={oracle['frac_certified_lookup']:.4f}", flush=True)

    os.makedirs(RES, exist_ok=True)
    for name, rows in (("p11v2_grid", grid_rows), ("p11v2_ablation", abl_rows),
                       ("p11v2_oracle", oracle_rows)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(RES, name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    print(f"grid={len(grid_rows)} ablation={len(abl_rows)} "
          f"matrices={len(oracle_rows)}")


if __name__ == "__main__":
    main()
