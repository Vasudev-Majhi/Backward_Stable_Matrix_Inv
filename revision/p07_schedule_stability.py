"""P0.7 / Review sec.5.2, sec.14, Reviewer A W7 -- MILP schedule stability.

CONCERN: "Model depth/width are outputs of a mixed-integer program. The solver's
tie-breaking is not guaranteed stable across versions or platforms; different
layer/head assignments give different R^L_i, R^U_i, hence different bits. The
'bitwise in the pinned environment' claim is therefore hostage to a MILP
solver's tie-breaking, which the paper never discusses."

DESIGN. For each circuit, compile it N times under DIFFERENT MILP solver
configurations (solver backend, random seed, thread count, time limit -- each of
which can change which optimal schedule is returned among ties). For each build
we record:
    - the schedule shape (n_layers, d_model)
    - a SHA-256 over the emitted weight tensors (byte identity of the program)
    - the slot map
and then EXECUTE each build on the same right-hand side and compare the
resulting solution vectors BIT FOR BIT.

This separates two questions the paper conflates:
    Q1  Is the emitted program byte-identical across schedules?   (probably not)
    Q2  Is the EXECUTED RESULT bit-identical across schedules?    (the claim
        that actually matters, and the one the paper should be making)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from craft_harness import (  # noqa: E402
    CompiledSolver, circuit_system, eta_rigal_gaches, host_lu_solve, load_circuits,
)

RES = os.path.join(HERE, "results")


def build_variant(cid, out_path, solver_kwargs):
    """Build one circuit with a specific MILP solver configuration.

    milp.py imports HiGHS / PULP_CBC_CMD from pulp *inside* the solve block and
    has no injection hook, so we patch the pulp namespace for the duration of
    the build. This is the least invasive way to vary the solver's tie-breaking.
    """
    import pulp
    import transformer_vm.scheduler.milp  # noqa: F401  (ensure module is loaded)

    backend = solver_kwargs.get("backend", "HiGHS")
    tl = solver_kwargs.get("timeLimit", 60)
    seed = solver_kwargs.get("seed", 0)
    threads = solver_kwargs.get("threads", 1)

    real_highs = pulp.HiGHS
    real_cbc = pulp.PULP_CBC_CMD

    def fake_highs(*a, **k):
        if backend != "HiGHS":
            class _Unavailable:
                def available(self):
                    return False
            return _Unavailable()
        # pulp's HiGHS forwards unknown kwargs straight to setOptionValue
        for kw in ({"random_seed": int(seed), "threads": int(threads)},
                   {"random_seed": int(seed)},
                   {}):
            try:
                return real_highs(msg=False, timeLimit=tl, **kw)
            except TypeError:
                continue
        return real_highs(msg=False, timeLimit=tl)

    def fake_cbc(*a, **k):
        opts = [f"randomSeed {seed}", f"randomCbcSeed {seed}"]
        return real_cbc(msg=0, timeLimit=tl, threads=threads, options=opts)

    pulp.HiGHS = fake_highs
    pulp.PULP_CBC_CMD = fake_cbc
    os.environ["MILP_TIME_LIMIT"] = str(tl)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from build_lu_direct import build_for_circuit
            build_for_circuit(cid, out_path=out_path)
    finally:
        pulp.HiGHS = real_highs
        pulp.PULP_CBC_CMD = real_cbc
    return out_path


def weight_hash(model_path):
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", nargs="+",
                    default=["CKT_0040", "CKT_0071", "CKT_0090", "CKT_0119"])
    args = ap.parse_args()

    variants = [
        {"name": "highs_seed0_t1", "backend": "HiGHS", "seed": 0, "threads": 1, "timeLimit": 60},
        {"name": "highs_seed1_t1", "backend": "HiGHS", "seed": 1, "threads": 1, "timeLimit": 60},
        {"name": "highs_seed7_t1", "backend": "HiGHS", "seed": 7, "threads": 1, "timeLimit": 60},
        {"name": "highs_seed0_t4", "backend": "HiGHS", "seed": 0, "threads": 4, "timeLimit": 60},
        {"name": "highs_tl10", "backend": "HiGHS", "seed": 0, "threads": 1, "timeLimit": 10},
        {"name": "cbc_default", "backend": "CBC", "threads": 1, "timeLimit": 60},
    ]

    circuits = load_circuits()
    rows = []
    for cid in args.circuits:
        c = circuits[cid]
        A, b, free, fixed, pc = circuit_system(c["Netlist"])
        bs = b * 1e4
        rng = np.random.default_rng(3)
        b_test = rng.normal(size=len(free))
        base_x = None
        for v in variants:
            d = os.path.join(HERE, "models_sched", v["name"])
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"model_{cid}_lu_direct.bin")
            try:
                if not os.path.exists(path + ".slots.json"):
                    build_variant(cid, path, v)
                side = json.load(open(path + ".slots.json"))
                S = CompiledSolver(path, c["Netlist"])
                r_src = S.solve(bs)
                r_rnd = S.solve(b_test)
                x_src = r_src["x"]; x_rnd = r_rnd["x"]
                nl = len(S.model.attn)
                dm = S.model.tok.weight.shape[1]
                row = dict(
                    circuit=cid, variant=v["name"], backend=v["backend"],
                    n_layers=nl, d_model=dm,
                    weight_sha256=weight_hash(path)[:16],
                    slot_x_value=side["slot_x_value"],
                    slot_x_new=side["slot_x_new"],
                    eta_src=eta_rigal_gaches(A, x_src, bs),
                    eta_rnd=eta_rigal_gaches(A, x_rnd, b_test),
                    x_src_hex=hashlib.sha256(x_src.tobytes()).hexdigest()[:16],
                    x_rnd_hex=hashlib.sha256(x_rnd.tobytes()).hexdigest()[:16],
                    bitwise_vs_host=bool(np.array_equal(x_rnd, host_lu_solve(A, b_test))),
                    pred_v=r_src["pred_v"],
                )
                if base_x is None:
                    base_x = (x_src.copy(), x_rnd.copy())
                    row["bitwise_vs_first_variant"] = True
                    row["max_ulp_diff_vs_first"] = 0.0
                else:
                    same = (np.array_equal(x_src, base_x[0])
                            and np.array_equal(x_rnd, base_x[1]))
                    row["bitwise_vs_first_variant"] = bool(same)
                    d1 = np.abs(x_rnd - base_x[1])
                    sc = np.maximum(np.abs(base_x[1]), 1e-300)
                    row["max_ulp_diff_vs_first"] = float(
                        np.max(d1 / sc) / (np.finfo(np.float64).eps))
                rows.append(row)
                print(f"{cid} {v['name']:>16}  L={nl} d={dm}  "
                      f"wsha={row['weight_sha256']}  "
                      f"bitwise_result={row['bitwise_vs_first_variant']}", flush=True)
            except Exception as e:
                print(f"{cid} {v['name']}: FAILED {type(e).__name__}: {e}", flush=True)

    os.makedirs(RES, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RES, "p07_schedule_stability.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    summ = {}
    for cid in args.circuits:
        sub = [r for r in rows if r["circuit"] == cid]
        if not sub:
            continue
        summ[cid] = {
            "n_variants_built": len(sub),
            "distinct_schedules": len({(r["n_layers"], r["d_model"]) for r in sub}),
            "distinct_weight_hashes": len({r["weight_sha256"] for r in sub}),
            "distinct_slot_maps": len({(r["slot_x_value"], r["slot_x_new"]) for r in sub}),
            "all_results_bit_identical": all(r["bitwise_vs_first_variant"] for r in sub),
            "max_ulp_diff": max(r["max_ulp_diff_vs_first"] for r in sub),
            "max_eta": max(max(r["eta_src"], r["eta_rnd"]) for r in sub),
        }
    out = {"per_circuit": summ, "variants": [v["name"] for v in variants]}
    out["verdict"] = (
        "Q1 (byte identity of the emitted program) and Q2 (bit identity of the "
        "executed result) are different claims; this table separates them."
    )
    with open(os.path.join(RES, "p07_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
