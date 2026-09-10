"""P0.1 / Review sec. 21 -- Run-time right-hand-side generalization.

CONCERN (Reviewer A W1, Reviewer B, sec.32.3): "T1 is never driven with any input
other than a column of the identity, and the Poisson diagnostic forms Xhat*1 on
the host instead of solving in-model. The hypothesis that T1 emits a
compile-time constant rather than executing a solve is not excluded by any
experiment in the paper."

HYPOTHESIS H1: T1 + runner computes a genuine function of a run-time input, and
its per-solve backward error for arbitrary b is statistically indistinguishable
from its error on identity columns.

DESIGN: one compiled model per matrix, NO recompilation between right-hand
sides. The runner hands b_i in at rhs_i exactly as it already hands y_i in at
bck_i (the declared runner contract, Table 6). RHS families:

  identity   -- columns e_j of I                  (the existing experiment)
  source     -- the physical source-driven RHS    (the paper's own b)
  gaussian   -- i.i.d. N(0,1)
  wide       -- components spanning 1e-8 .. 1e8 in magnitude
  ones       -- the all-ones Poisson forcing vector, SOLVED IN-MODEL
  adversarial-- aligned with the smallest singular direction of A

DEPENDENT VARIABLES: per-solve eta (Rigal-Gaches), relative forward error
against a 160-digit mpmath reference, agreement with the host binary64
recurrence, and selector hit rate against the float64 production trace.

Statistical unit: the (matrix, right-hand side) pair. We report per-matrix
maxima and the max-of-max, as the paper already does correctly -- no
inferential statistics, no rates on non-sample populations.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from craft_harness import (  # noqa: E402
    CompiledSolver, build_model_for, circuit_system, eta_rigal_gaches,
    growth_factor, host_lu_solve, load_circuits, reference_solve,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def rhs_families(A, b_source, n, rng, n_random):
    fams = []
    for j in range(min(n, 8)):
        e = np.zeros(n); e[j] = 1.0
        fams.append((f"identity_e{j}", e))
    fams.append(("source", b_source * 1e4))          # scaled units (SCALE=1e4)
    fams.append(("ones", np.ones(n)))                # Poisson forcing, in-model
    for k in range(n_random):
        fams.append((f"gaussian_{k}", rng.normal(size=n)))
    for k in range(n_random):
        mag = 10.0 ** rng.uniform(-8, 8, size=n)
        fams.append((f"wide_{k}", rng.normal(size=n) * mag))
    if n > 0:
        try:
            U, S, Vt = np.linalg.svd(A)
            fams.append(("adversarial_minsv", Vt[-1] * np.linalg.norm(b_source * 1e4)))
        except np.linalg.LinAlgError:
            pass
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=40)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--min-free", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(RES, "p01_runtime_rhs.csv"))
    args = ap.parse_args()

    circuits = load_circuits()
    rng = np.random.default_rng(20260910)
    rows = []
    picked = []
    skipped_fixed_target = []
    for cid, c in circuits.items():
        A, b, free, fixed, pc = circuit_system(c["Netlist"])
        if pc.is_fixed[int(c["Target_Node"])]:
            # The compiler emits a direct source read for these; the LU path is
            # built but never exercised, so they cannot inform a claim about
            # execution. Reported separately (see p01b_fixed_target.py).
            skipped_fixed_target.append(cid)
            continue
        if len(free) >= args.min_free:
            picked.append((cid, c, A, b, free))
    # spread over the size range rather than taking the first k
    picked.sort(key=lambda t: len(t[4]))
    if len(picked) > args.n_circuits:
        idx = np.linspace(0, len(picked) - 1, args.n_circuits).astype(int)
        picked = [picked[i] for i in idx]

    for ci, (cid, c, A, b, free) in enumerate(picked):
        n = len(free)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                mp_ = build_model_for(cid, os.path.join(HERE, "models"))
                S = CompiledSolver(mp_, c["Netlist"])
        except Exception as e:
            print(f"[skip] {cid}: {type(e).__name__}: {e}")
            continue

        # float64 production reference trace for this matrix (source RHS)
        ref = S.solve(b * 1e4, record=True)
        ref_sel = list(ref["cache"].sel)
        kappa = float(np.linalg.cond(A)) if n else float("nan")
        gf = growth_factor(A)

        for name, bx in rhs_families(A, b, n, rng, args.n_random):
            r = S.solve(bx, record=True)
            x = r["x"]
            eta = eta_rigal_gaches(A, x, bx)
            xh = host_lu_solve(A, bx)
            xr = reference_solve(A, bx)
            den = max(np.max(np.abs(xr)), 1e-300)
            fwd = float(np.max(np.abs(x - xr)) / den)
            host_agree = float(np.max(np.abs(x - xh)) / max(np.max(np.abs(xh)), 1e-300))
            bitwise = bool(np.array_equal(x, xh))
            sel = r["cache"].sel
            m = min(len(sel), len(ref_sel))
            hit = float(np.mean([sel[i] == ref_sel[i] for i in range(m)])) if m else float("nan")
            rows.append(dict(
                circuit=cid, n_free=n, kappa=kappa, growth=gf, rhs=name,
                family=name.split("_")[0], eta=eta, fwd_rel_err=fwd,
                host_rel_diff=host_agree, bitwise_host=bitwise,
                selector_hit=hit, n_lookups=len(sel),
                bnorm=float(np.max(np.abs(bx))),
                xnorm=float(np.max(np.abs(x))),
            ))
        print(f"[{ci+1}/{len(picked)}] {cid} n={n} kappa={kappa:.2e} "
              f"max_eta={max(r['eta'] for r in rows if r['circuit']==cid):.3e}")

    os.makedirs(RES, exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summary -------------------------------------------------------
    fams = sorted({r["family"] for r in rows})
    summary = {"n_matrices": len({r["circuit"] for r in rows}),
               "n_solves": len(rows), "by_family": {}}
    for fam in fams:
        sub = [r for r in rows if r["family"] == fam]
        summary["by_family"][fam] = {
            "n_solves": len(sub),
            "max_eta": max(r["eta"] for r in sub),
            "median_eta": float(np.median([r["eta"] for r in sub])),
            "max_fwd_rel_err": max(r["fwd_rel_err"] for r in sub),
            "max_host_rel_diff": max(r["host_rel_diff"] for r in sub),
            "frac_bitwise_host": float(np.mean([r["bitwise_host"] for r in sub])),
            "min_selector_hit": min(r["selector_hit"] for r in sub),
        }
    summary["max_of_max_eta"] = max(r["eta"] for r in rows)
    summary["excluded_fixed_target_circuits"] = skipped_fixed_target
    summary["n_excluded_fixed_target"] = len(skipped_fixed_target)
    summary["u"] = float(np.finfo(np.float64).eps / 2)
    with open(os.path.join(RES, "p01_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
