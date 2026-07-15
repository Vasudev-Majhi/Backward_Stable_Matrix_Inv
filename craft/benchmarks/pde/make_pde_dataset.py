"""Bundle PDE benchmark problems into a JSONL with the same schema as
circuit_dataset_rv.jsonl (so the existing build/runner pipeline can consume it).

Usage:
    python benchmarks/pde/make_pde_dataset.py [--out path]
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import argparse
import json
import os

from benchmarks.pde.heat_2d import heat_problems
from benchmarks.pde.poisson_2d import poisson_problems

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), "results", "pde_dataset.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--include-poisson", action="store_true",
                    help="Include Poisson problems (require I-source extension; default: include)")
    ap.add_argument("--no-poisson", dest="include_poisson", action="store_false")
    ap.set_defaults(include_poisson=True)
    args = ap.parse_args()

    problems = []
    problems.extend(heat_problems())
    if args.include_poisson:
        problems.extend(poisson_problems())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for p in problems:
            # Strip non-schema metadata (family, rows, cols, etc.) into a side dict
            # but keep the main keys matching circuit_dataset_rv.jsonl.
            json.dump(p, f)
            f.write("\n")

    n_heat = sum(1 for p in problems if p.get("family") == "heat_2d")
    n_pois = sum(1 for p in problems if p.get("family") == "poisson_2d")
    print(f"Wrote {len(problems)} problems  (heat={n_heat}, poisson={n_pois})  ->  {args.out}")


if __name__ == "__main__":
    main()
