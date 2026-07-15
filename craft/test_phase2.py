"""Phase 2+3 smoke test: run the DSL graph through the evaluator on CKT_0001.

Interleaves the fixed token stream with model-predicted v-tokens
(runner-style). Compares the final predicted voltage against the Python
Jacobi reference.
"""
from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401

from interpreter import CircuitMachine, V_STEP
from jacobi_reference import jacobi_solve_parsed
from parse import parse_netlist
from tokenize_netlist import tokenize, PREDICTED
from fast_attention import make_fast_runtime


def run_one(netlist: str, target_node: int, T: int, verbose: bool = False):
    pc = parse_netlist(netlist)
    machine = CircuitMachine(pc, target_node, T)
    pg, meta = machine.build()

    rt = make_fast_runtime(pg)

    fixed_toks, _, _ = tokenize(netlist, target_node, T=T)
    predicted_log: list[str] = []
    last_vals = None

    i = 0
    while i < len(fixed_toks):
        tok = fixed_toks[i]
        if tok == PREDICTED:
            # Use the PREVIOUS step's vals to predict this position's token
            # (predict_next scores next-position candidates from current vals).
            # But we need the vals AT the previous position (already produced
            # by last_vals when we stepped through tok i-1).
            predicted = rt.predict_next(last_vals)
            predicted_log.append(predicted)
            if verbose and i < 30:
                print(f"  pos {rt.pos}: predicted {predicted}")
            last_vals = rt.step(predicted)
        else:
            last_vals = rt.step(tok)
            if verbose and i < 30:
                print(f"  pos {rt.pos - 1}: fed    {tok}")
        i += 1

    rt.destroy()

    # The LAST predicted token should be the readout's v-emission.
    final_vk = predicted_log[-1]  # e.g. "v_240"
    assert final_vk.startswith("v_"), f"expected v_<k>, got {final_vk}"
    k = int(final_vk.split("_")[1])
    predicted_voltage = k * V_STEP / 10000.0

    # Reference Jacobi
    ref_voltage = jacobi_solve_parsed(pc, target_node, T)

    return predicted_voltage, ref_voltage, predicted_log


def main():
    import time
    path = "dataset/circuit_dataset_rv.jsonl"

    # Usage: python test_phase2.py [ids_csv_or_limit] [T]
    # If first arg is a comma-list of IDs, run those. Otherwise treat as limit.
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "3"
    T_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    by_id: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            by_id[c["ID"]] = c

    if "CKT_" in arg1:
        ids = arg1.split(",")
    else:
        limit = int(arg1)
        ids = list(by_id.keys())[:limit]

    passed = 0
    for cid in ids:
        c = by_id[cid]
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        truth = c["Ground_Truth_Vout"]

        t0 = time.time()
        try:
            pred, ref, _log = run_one(netlist, target, T_arg, verbose=False)
            elapsed = time.time() - t0
            err = abs(pred - truth)
            ok = err < 0.05
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"{status} {cid:<10} pred={pred:>8.4f} truth={truth:>8.4f} ref={ref:>8.4f} "
                  f"err={err:.4f} time={elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR {cid} ({elapsed:.1f}s): {e}")
    print(f"\n{passed}/{len(ids)} passed (tol 0.05V)")


if __name__ == "__main__":
    main()
