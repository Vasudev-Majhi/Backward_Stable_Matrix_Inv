"""Step 4 gate: lu_direct_reference must match direct_reference per-circuit.

Runs the entire 154-circuit dataset through both solvers and asserts the raw
unquantized x-values agree to 1e-9. Failure here means a bug in lu_factor or
the partition logic, NOT in any DSL code.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))  # lu_pipeline/

import _path  # noqa: F401, E402

import numpy as np  # noqa: E402

from parse import parse_netlist  # type: ignore  # noqa: E402

from lu_direct_reference import build_partition  # noqa: E402
from lu_factor import back_sub, doolittle, forward_sub  # noqa: E402

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    os.path.join(PROJECT, "dataset", "circuit_dataset_rv.jsonl"),
)


def main() -> int:
    if not os.path.exists(DATASET_PATH):
        print(f"FAIL: dataset not found at {DATASET_PATH}")
        return 2

    n_total = n_pass = n_fail = 0
    max_diff = 0.0
    failures: list[tuple[str, float]] = []

    with open(DATASET_PATH) as f:
        for line in f:
            c = json.loads(line)
            cid = c["ID"]
            netlist = c["Netlist"]
            target = int(c["Target_Node"])
            pc = parse_netlist(netlist)
            n_total += 1

            if pc.is_fixed[target]:
                continue  # degenerate; no linear solve to compare

            free, fixed, A_FF, A_FP = build_partition(pc)
            if not free:
                continue
            v_P = np.array([pc.fixed_voltage[fn] for fn in fixed], dtype=np.float64)
            b = -A_FP @ v_P

            x_solve = np.linalg.solve(A_FF, b)

            L, U = doolittle(A_FF)
            y = forward_sub(L, b)
            x_lu = back_sub(U, y)

            diff = float(np.max(np.abs(x_lu - x_solve)))
            if diff > max_diff:
                max_diff = diff
            if diff < 1e-9:
                n_pass += 1
            else:
                n_fail += 1
                failures.append((cid, diff))

    print(f"reference parity: {n_pass}/{n_total - (n_total - n_pass - n_fail)} compared, "
          f"{n_fail} fail, max_diff={max_diff:.2e}")
    if failures:
        for cid, d in failures[:10]:
            print(f"  FAIL {cid}: max_diff={d:.2e}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
