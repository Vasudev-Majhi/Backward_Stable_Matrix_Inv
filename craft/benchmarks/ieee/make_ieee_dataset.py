"""Bundle IEEE DC power-flow problems into a JSONL.

Schema mirrors circuit_dataset_rv.jsonl so the existing build/runner pipeline
can consume it after the I-source extension is loaded.

Usage:
    python benchmarks/ieee/make_ieee_dataset.py [--out path]
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import argparse
import json
import os

from benchmarks.ieee.ieee_cases import ALL_CASES
from benchmarks.ieee.ieee_groundtruth import dc_pf_solve
from benchmarks.ieee.ieee_to_netlist import dc_pf_to_netlist

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), "results", "ieee_dataset.jsonl")


def make_problems() -> list[dict]:
    """Generate IEEE problems. Vary V_bias / K / target bus to get a few problems
    per case so the benchmark has more than 2 data points."""
    problems: list[dict] = []
    pid = 0

    for case_key, case in ALL_CASES.items():
        bus_ids = [b[0] for b in case["buses"]]
        slack = case["slack_bus"]
        non_slack = [b for b in bus_ids if b != slack]

        # Pick a few target buses for variety: max-id, mid-id, near-slack.
        target_buses = [
            max(non_slack),
            non_slack[len(non_slack) // 2],
            non_slack[2] if len(non_slack) > 2 else non_slack[0],
        ]
        target_buses = list(dict.fromkeys(target_buses))  # dedupe, keep order

        # Vary scaling. K=30 keeps angles in the [4-20]V range typically.
        # V_bias=12 keeps everything well away from boundaries.
        scaling_variants = [
            (12.0, 30.0),
            (12.0, 20.0),    # smaller scaling = smaller angle deviations from V_bias
        ]

        for tb in target_buses:
            for V_bias, K in scaling_variants:
                netlist, target_node, meta = dc_pf_to_netlist(
                    case, V_bias=V_bias, K=K, target_bus=tb,
                )
                v_target, _ = dc_pf_solve(case, V_bias=V_bias, K=K, target_bus=tb)

                pid += 1
                # Skip if ground truth is outside the safe voltage range
                # (something went wrong with scaling).
                if not (0.5 <= v_target <= 23.5):
                    print(f"  WARN: skipping IEEE_{case['name']} tb={tb} V_bias={V_bias} K={K}: "
                          f"ground truth {v_target:.4f} outside safe range")
                    continue

                problems.append({
                    "ID": f"IEEE_{case_key.upper()}_{pid:04d}",
                    "Description": (
                        f"DC power flow {case['name']} target_bus={tb} "
                        f"V_bias={V_bias} K={K}"
                    ),
                    "Netlist": netlist,
                    "Target_Node": str(target_node),
                    "Ground_Truth_Vout": float(v_target),
                    "Complexity": "Hard" if case_key == "case30" else "Intermediate",
                    "family": "ieee_dc_pf",
                    "case": case_key,
                    "target_bus": tb,
                    "V_bias": V_bias,
                    "K": K,
                    "n_buses": meta["n_buses"],
                    "n_branches": meta["n_branches"],
                    "n_isources": meta["n_isources"],
                })

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    problems = make_problems()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for p in problems:
            json.dump(p, f)
            f.write("\n")

    by_case: dict[str, int] = {}
    for p in problems:
        by_case[p["case"]] = by_case.get(p["case"], 0) + 1
    print(f"Wrote {len(problems)} IEEE problems  (per case: {by_case})  ->  {args.out}")


if __name__ == "__main__":
    main()
