"""Run PDE benchmark sweep with Jacobi + Hull KV."""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import argparse
import os

from benchmarks.orchestrators._helper import RESULTS_DIR, orchestrate

DEFAULT_CSV = os.path.join(RESULTS_DIR, "pde_jacobi.csv")


def main():
    ap = argparse.ArgumentParser(description="PDE benchmark - Jacobi + Hull KV")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="Run only the first problem")
    ap.add_argument("--no-hull", action="store_true", help="Use StandardKVCache")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--v-step", type=int, default=None,
                    help="Quantization step in scaled units (500=0.05V default; 100=0.01V)")
    ap.add_argument("--k-levels", type=int, default=None,
                    help="Number of v_k tokens (480 default at 0.05V; 2400 at 0.01V)")
    ap.add_argument("--suffix", type=str, default="",
                    help="Model bin path suffix (e.g. _v01) to keep V_STEP variants distinct")
    args = ap.parse_args()

    orchestrate(
        benchmark="pde", algorithm="jacobi", out_csv=args.csv,
        use_hull=not args.no_hull, limit=args.limit, smoke=args.smoke,
        tol=args.tol, force_rebuild=args.force_rebuild,
        v_step=args.v_step, k_levels=args.k_levels, model_suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
