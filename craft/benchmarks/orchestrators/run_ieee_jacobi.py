"""Run IEEE DC power-flow benchmark sweep with Jacobi + Hull KV."""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import argparse
import os

from benchmarks.orchestrators._helper import RESULTS_DIR, orchestrate

DEFAULT_CSV = os.path.join(RESULTS_DIR, "ieee_jacobi.csv")


def main():
    ap = argparse.ArgumentParser(description="IEEE DC PF benchmark - Jacobi + Hull KV")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-hull", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    orchestrate(
        benchmark="ieee", algorithm="jacobi", out_csv=args.csv,
        use_hull=not args.no_hull, limit=args.limit, smoke=args.smoke,
        tol=args.tol, force_rebuild=args.force_rebuild,
        v_step=args.v_step, k_levels=args.k_levels, model_suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
