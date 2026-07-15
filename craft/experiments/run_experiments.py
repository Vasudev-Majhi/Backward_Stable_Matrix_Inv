"""Main experiment orchestrator — runs E1..E11 (E12 dropped).

Designed for fully autonomous execution on the remote server under tmux.
Writes incremental CSV checkpoints after every circuit so partial results
survive crashes / kills. At the end, invokes make_figures.py to produce
all paper figures + summary.md.

Outputs (all under `results/`):
  results_main.csv        — one row per circuit (E1-E5, E7-E10)
  convergence_E3.csv      — error vs T on 5 sampled circuits
  heat_diffusion_E11.csv  — per-grid-size pred vs analytical
  build_stats.csv         — E7/E8 model build stats
  figures/*.png           — E6 phase diagram + all other plots
  summary.md              — human-readable summary

Usage:
  python run_experiments.py                # full run
  python run_experiments.py --limit 10     # first 10 circuits only (sanity)
  python run_experiments.py --skip-build   # assume model_*.bin already built
  python run_experiments.py --skip-transformer-sample  # DSL only, no real-transformer sample
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import logging
import os
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("experiments")

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

MAIN_COLUMNS = [
    "circuit_id", "complexity", "N", "num_resistors", "num_fixed_nodes",
    "T_used", "pred_V_dsl", "truth_V", "abs_error", "pass_fail",
    "pred_V_transformer", "transformer_matches_dsl",
    "jacobi_ref_V", "direct_solve_V",
    "rho_M", "kappa_A",
    "seq_length", "dsl_time_ms", "transformer_time_ms",
    "n_layers", "d_model", "d_ffn", "vocab", "model_size_bytes",
    "error",
]

SAMPLE_FOR_E2 = [
    # 2 Basic
    "CKT_0001", "CKT_0023",
    # 3 Intermediate
    "CKT_0040", "CKT_0045", "CKT_0050",
    # 5 Hard
    "CKT_0100", "CKT_0135", "CKT_0140", "CKT_0145", "CKT_0150",
]

# E3 convergence sweep: 7 circuits spanning ρ from ~0.35 to ~0.9999,
# 11 log-spaced T values.
SAMPLE_FOR_E3 = ["CKT_0020", "CKT_0022", "CKT_0070", "CKT_0067",
                 "CKT_0065", "CKT_0058", "CKT_0133"]
E3_T_VALUES = [1, 2, 3, 5, 8, 13, 20, 35, 50, 100, 200]

HEAT_GRID_SIZES = [(3, 3), (5, 5), (10, 10)]


def load_circuits() -> list[dict]:
    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    return circuits


def pick_T(N: int) -> int:
    """T = max(1000, 50·N) — enough iterations to reach quantization floor
    on every circuit, regardless of tier. The earlier T=200 policy for Hard
    was wrong — it under-iterated mid-ρ Hard circuits that converge in
    ~O(1/(1-ρ)) iterations, which for ρ≈0.99 is ~500."""
    return max(1000, 50 * N)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── E1 + E2 + E3 + E4 + E5 + E7 + E8 + E9 + E10 master loop ──────────

def run_main_loop(
    circuits: list[dict],
    *,
    run_transformer_sample: bool = True,
    transformer_sample_ids: set[str] | None = None,
    main_csv_path: str | None = None,
) -> list[dict]:
    """Loop over circuits, compute everything, checkpoint incrementally."""
    from dsl_runner import run_one_dsl
    from jacobi_reference import jacobi_solve_parsed
    from parse import parse_netlist
    from spectrum import compute_rho_kappa, direct_solve

    if transformer_sample_ids is None:
        transformer_sample_ids = set(SAMPLE_FOR_E2)

    rows: list[dict] = []
    sample_transformer = None  # lazy import

    for i, c in enumerate(circuits):
        cid = c["ID"]
        row = {col: "" for col in MAIN_COLUMNS}
        row["circuit_id"] = cid
        row["complexity"] = c.get("Complexity", "")
        row["truth_V"] = c["Ground_Truth_Vout"]

        try:
            pc = parse_netlist(c["Netlist"])
            target = int(c["Target_Node"])
            T = pick_T(pc.num_nodes)
            row["N"] = pc.num_nodes
            row["num_resistors"] = len(pc.resistors)
            row["num_fixed_nodes"] = sum(pc.is_fixed)
            row["T_used"] = T

            # E4, E5 — spectrum + direct solve
            try:
                rho, kappa = compute_rho_kappa(pc)
                row["rho_M"] = f"{rho:.10f}"
                row["kappa_A"] = f"{kappa:.6e}"
            except Exception as e:  # noqa: BLE001
                row["rho_M"] = "inf"
                row["kappa_A"] = "inf"
                log.warning(f"{cid}: spectrum failed: {e}")

            try:
                row["direct_solve_V"] = f"{direct_solve(pc, target):.6f}"
            except Exception as e:  # noqa: BLE001
                row["direct_solve_V"] = ""
                log.warning(f"{cid}: direct solve failed: {e}")

            # E1 — DSL evaluator (main result)
            dsl_res = run_one_dsl(c["Netlist"], target, T, parsed=pc)
            row["pred_V_dsl"] = f"{dsl_res['pred_V']:.4f}"
            row["seq_length"] = dsl_res["seq_length"]
            row["dsl_time_ms"] = f"{dsl_res['runtime_s']*1000:.1f}"

            abs_err = abs(dsl_res["pred_V"] - float(c["Ground_Truth_Vout"]))
            row["abs_error"] = f"{abs_err:.4f}"
            row["pass_fail"] = "PASS" if abs_err <= 0.05 else "FAIL"

            # Jacobi reference at same T (bit-identical to DSL — sanity check)
            try:
                ref = jacobi_solve_parsed(pc, target, T)
                row["jacobi_ref_V"] = f"{ref:.4f}"
            except Exception as e:  # noqa: BLE001
                row["jacobi_ref_V"] = ""
                log.warning(f"{cid}: jacobi ref failed: {e}")

            # E7/E8 — MILP/model stats (re-read model.bin if it exists)
            model_path = os.path.join(REPO_ROOT, f"model_{cid}.bin")
            if os.path.exists(model_path):
                row["model_size_bytes"] = os.path.getsize(model_path)
                try:
                    from transformer_vm.model.weights import load_weights
                    m, all_toks, _ = load_weights(model_path)
                    row["n_layers"] = len(m.attn)
                    row["d_model"] = m.tok.weight.shape[1]
                    row["d_ffn"] = m.ff_in[0].weight.shape[0] // 2
                    row["vocab"] = len(all_toks)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"{cid}: couldn't read model_{cid}.bin: {e}")

            # E2 + E10 sample — real transformer on 10 spot-check circuits.
            # Uses a small T (50) because StandardKVCache is O(S²) and real T
            # would take hours per circuit. E2 just validates transformer ≡ DSL
            # (correctness), which is T-independent. E10 timing is measured at
            # this small T as the per-circuit transformer infer_runtime_s.
            # The DSL comparison uses the SAME small T so transformer_matches_dsl
            # is meaningful.
            if (
                run_transformer_sample
                and cid in transformer_sample_ids
                and os.path.exists(model_path)
            ):
                if sample_transformer is None:
                    from cpu_runner import run_one_transformer
                    sample_transformer = run_one_transformer
                try:
                    T_sample = 50
                    # Run DSL at the same small T for a bit-for-bit comparison.
                    from dsl_runner import run_one_dsl as _rod
                    dsl_sample = _rod(c["Netlist"], target, T_sample, parsed=pc)
                    tres = sample_transformer(
                        model_path, c["Netlist"], target, T_sample, cache="standard"
                    )
                    row["pred_V_transformer"] = f"{tres['pred_V']:.4f}"
                    row["transformer_time_ms"] = f"{tres['infer_runtime_s']*1000:.1f}"
                    row["transformer_matches_dsl"] = (
                        abs(tres["pred_V"] - dsl_sample["pred_V"]) < 1e-6
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning(f"{cid}: transformer sample failed: {e}")
                    row["pred_V_transformer"] = ""
                    row["transformer_matches_dsl"] = ""

        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            log.error(f"{cid}: {row['error']}")
            log.debug(traceback.format_exc())

        rows.append(row)

        # Incremental checkpoint
        if main_csv_path:
            write_csv(main_csv_path, rows, MAIN_COLUMNS)

        if (i + 1) % 5 == 0 or (i + 1) == len(circuits):
            passed = sum(1 for r in rows if r["pass_fail"] == "PASS")
            log.info(
                f"[{i+1}/{len(circuits)}] {cid} "
                f"pred={row.get('pred_V_dsl', '?')} truth={row['truth_V']} "
                f"{row.get('pass_fail', '')} | running pass={passed}"
            )

    return rows


# ── E3 — error vs T convergence sweep ──────────────────────────────────

def run_e3_convergence(circuits_by_id: dict[str, dict]) -> list[dict]:
    from dsl_runner import run_one_dsl
    from parse import parse_netlist

    rows = []
    for cid in SAMPLE_FOR_E3:
        if cid not in circuits_by_id:
            log.warning(f"E3: {cid} not in dataset, skipping")
            continue
        c = circuits_by_id[cid]
        pc = parse_netlist(c["Netlist"])
        target = int(c["Target_Node"])
        truth = float(c["Ground_Truth_Vout"])

        for T in E3_T_VALUES:
            try:
                r = run_one_dsl(c["Netlist"], target, T, parsed=pc)
                err = abs(r["pred_V"] - truth)
                rows.append({
                    "circuit_id": cid,
                    "complexity": c.get("Complexity", ""),
                    "T": T,
                    "pred_V": f"{r['pred_V']:.4f}",
                    "truth_V": truth,
                    "abs_error": f"{err:.6f}",
                    "seq_length": r["seq_length"],
                    "runtime_s": f"{r['runtime_s']:.3f}",
                })
                log.info(f"E3 {cid} T={T:>3} pred={r['pred_V']:.4f} err={err:.4f}")
            except Exception as e:  # noqa: BLE001
                log.error(f"E3 {cid} T={T}: {e}")
                rows.append({
                    "circuit_id": cid, "T": T, "abs_error": "",
                    "error": f"{type(e).__name__}: {e}",
                })
    return rows


# ── E11 — heat diffusion generalization ────────────────────────────────

def run_e11_heat() -> list[dict]:
    """E11: run heat-diffusion grids through the full pipeline.

    Uses ASYMMETRIC boundary conditions (top=9.3V, bottom=1.7V, left=2.9V,
    right=5.1V) to avoid quantization half-step ties that cause banker's
    rounding to lock DSL output at a biased fixed point. The DSL is a
    quantized Jacobi, so the correct baseline for "correctness" is the
    Python Jacobi reference, not the continuous analytical solution.
    """
    from dsl_runner import run_one_dsl
    from heat_netlist import thermal_grid_to_spice
    from jacobi_reference import _auto_T, jacobi_solve_parsed
    from parse import parse_netlist

    # Asymmetric, realistic BCs. Analytical lands near (but not exactly on)
    # V_STEP half-steps, so quantization error is the dominant error source
    # rather than banker's-rounding ties.
    BCs = dict(top_temp=10.0, bottom_temp=0.0, left_temp=4.0, right_temp=7.0)
    # Pass tolerance: 2 V_STEPs = 0.1V — compounded quantization floor for
    # multi-node Jacobi at V_STEP=0.05V. Tighter tolerance demands finer
    # quantization (more v_k tokens), which is out of scope for v1.
    E11_TOL_V = 0.1

    rows = []
    for rows_n, cols_n in HEAT_GRID_SIZES:
        try:
            netlist, target, analytical = thermal_grid_to_spice(
                rows_n, cols_n, conductivity=1.0, **BCs,
            )
            pc = parse_netlist(netlist)
            T = _auto_T(pc.num_nodes)
            jacobi_ref = jacobi_solve_parsed(pc, target, T)
            r = run_one_dsl(netlist, target, T, parsed=pc)

            err_vs_jacobi = abs(r["pred_V"] - jacobi_ref)
            err_vs_analytical = abs(r["pred_V"] - analytical)
            rows.append({
                "grid": f"{rows_n}x{cols_n}",
                "N": pc.num_nodes,
                "T": T,
                "pred_V": f"{r['pred_V']:.4f}",
                "jacobi_ref_V": f"{jacobi_ref:.4f}",
                "analytical_V": f"{analytical:.4f}",
                "err_vs_jacobi": f"{err_vs_jacobi:.6f}",
                "err_vs_analytical": f"{err_vs_analytical:.6f}",
                "seq_length": r["seq_length"],
                "runtime_s": f"{r['runtime_s']:.2f}",
                "pass": err_vs_jacobi <= E11_TOL_V,
                "tol_V": E11_TOL_V,
            })
            log.info(
                f"E11 {rows_n}x{cols_n}: pred={r['pred_V']:.4f} "
                f"jacobi_ref={jacobi_ref:.4f} analytical={analytical:.4f} "
                f"err_vs_jacobi={err_vs_jacobi:.4f}"
            )
        except Exception as e:  # noqa: BLE001
            log.error(f"E11 {rows_n}x{cols_n}: {e}")
            log.debug(traceback.format_exc())
            rows.append({
                "grid": f"{rows_n}x{cols_n}",
                "error": f"{type(e).__name__}: {e}",
            })
    return rows


# ── Orchestration ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N circuits (sanity testing)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build_all (assume model_*.bin already exist)")
    parser.add_argument("--skip-transformer-sample", action="store_true",
                        help="Skip real-transformer spot-check on SAMPLE_FOR_E2")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation at the end")
    parser.add_argument("--skip-heat", action="store_true",
                        help="Skip E11 heat diffusion")
    parser.add_argument("--skip-main", action="store_true",
                        help="Skip Phase B main loop (E1-E2-E4-E5-E7-E8-E9-E10)")
    parser.add_argument("--skip-e3", action="store_true",
                        help="Skip E3 convergence sweep")
    parser.add_argument("--skip-e12", action="store_true",
                        help="Skip E12 V_STEP ablation")
    parser.add_argument("--skip-e13", action="store_true",
                        help="Skip E13 T-scaling + theoretical overlay")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_start = time.time()
    log.info(f"Starting experiments (dataset: {DATASET_PATH})")

    circuits = load_circuits()
    if args.limit:
        circuits = circuits[:args.limit]
    log.info(f"Loaded {len(circuits)} circuits")
    circuits_by_id = {c["ID"]: c for c in circuits}

    # Phase A — build all models (E7, E8) unless --skip-build
    if not args.skip_build:
        log.info("=== Phase A: build_all ===")
        try:
            import build_all
            build_all.build_all(limit=args.limit)
        except Exception as e:  # noqa: BLE001
            log.error(f"build_all failed: {e}")
            log.debug(traceback.format_exc())

    # Phase B — main loop (E1, E2, E4, E5, E7, E8, E9, E10)
    if args.skip_main:
        log.info("=== Phase B: skipped (--skip-main) ===")
    else:
        log.info("=== Phase B: main loop ===")
        main_csv = os.path.join(RESULTS_DIR, "results_main.csv")
        run_main_loop(
            circuits,
            run_transformer_sample=not args.skip_transformer_sample,
            main_csv_path=main_csv,
        )

    # Phase C — E3 convergence sweep
    if not args.skip_e3:
        log.info("=== Phase C: E3 convergence sweep ===")
        try:
            e3_rows = run_e3_convergence(circuits_by_id)
            e3_csv = os.path.join(RESULTS_DIR, "convergence_E3.csv")
            e3_fields = [
                "circuit_id", "complexity", "T", "pred_V", "truth_V",
                "abs_error", "seq_length", "runtime_s", "error",
            ]
            write_csv(e3_csv, e3_rows, e3_fields)
            log.info(f"E3: wrote {e3_csv} ({len(e3_rows)} rows)")
        except Exception as e:  # noqa: BLE001
            log.error(f"E3 failed: {e}")
            log.debug(traceback.format_exc())

    # Phase D — E11 heat diffusion
    if not args.skip_heat:
        log.info("=== Phase D: E11 heat diffusion ===")
        try:
            e11_rows = run_e11_heat()
            e11_csv = os.path.join(RESULTS_DIR, "heat_diffusion_E11.csv")
            e11_fields = [
                "grid", "N", "T", "pred_V", "jacobi_ref_V", "analytical_V",
                "err_vs_jacobi", "err_vs_analytical",
                "seq_length", "runtime_s", "pass", "tol_V", "error",
            ]
            write_csv(e11_csv, e11_rows, e11_fields)
            log.info(f"E11: wrote {e11_csv} ({len(e11_rows)} rows)")
        except Exception as e:  # noqa: BLE001
            log.error(f"E11 failed: {e}")
            log.debug(traceback.format_exc())

    # Phase F — E12 V_STEP ablation (on 10 Mode B circuits)
    if not args.skip_e12:
        log.info("=== Phase F: E12 V_STEP ablation ===")
        try:
            import e12_vstep
            e12_vstep.run_e12()
        except Exception as e:  # noqa: BLE001
            log.error(f"E12 failed: {e}")
            log.debug(traceback.format_exc())

    # Phase G — E13 T-scaling with theoretical overlay
    if not args.skip_e13:
        log.info("=== Phase G: E13 T-scaling ===")
        try:
            import e13_tscaling
            e13_tscaling.run_e13()
        except Exception as e:  # noqa: BLE001
            log.error(f"E13 failed: {e}")
            log.debug(traceback.format_exc())

    # Phase E — figures
    if not args.skip_figures:
        log.info("=== Phase E: figures ===")
        try:
            import make_figures
            make_figures.make_all()
        except Exception as e:  # noqa: BLE001
            log.error(f"make_figures failed: {e}")
            log.debug(traceback.format_exc())

    # Final summary
    try:
        write_summary()
    except Exception as e:  # noqa: BLE001
        log.error(f"write_summary failed: {e}")

    elapsed = time.time() - t_start
    log.info(f"=== ALL DONE in {elapsed/60:.1f} min ===")


def write_summary():
    """Generate summary.md from results_main.csv."""
    main_csv = os.path.join(RESULTS_DIR, "results_main.csv")
    if not os.path.exists(main_csv):
        return
    with open(main_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_tier: dict[str, list] = {}
    for r in rows:
        by_tier.setdefault(r["complexity"] or "Unknown", []).append(r)

    lines = ["# Experiment Summary", ""]
    lines.append(f"Total circuits: {len(rows)}")
    lines.append("")
    lines.append("## Accuracy per tier (DSL evaluator, T per tier)")
    lines.append("")
    lines.append("| Tier | N | Pass | Fail | Pass rate |")
    lines.append("|------|---|------|------|-----------|")
    total_pass = 0
    for tier in ("Basic", "Intermediate", "Hard"):
        tier_rows = by_tier.get(tier, [])
        n = len(tier_rows)
        p = sum(1 for r in tier_rows if r["pass_fail"] == "PASS")
        f = n - p
        total_pass += p
        rate = f"{p/n*100:.1f}%" if n else "—"
        lines.append(f"| {tier} | {n} | {p} | {f} | {rate} |")
    lines.append(f"| **Total** | {len(rows)} | {total_pass} | {len(rows)-total_pass} | "
                 f"{total_pass/len(rows)*100:.1f}% |")
    lines.append("")

    # MILP universality check (E7)
    layer_set = {r["n_layers"] for r in rows if r["n_layers"]}
    dmodel_set = {r["d_model"] for r in rows if r["d_model"]}
    lines.append("## MILP universality (E7)")
    lines.append("")
    lines.append(f"- Unique n_layers values across all circuits: `{sorted(layer_set)}`")
    lines.append(f"- Unique d_model values across all circuits: `{sorted(dmodel_set)}`")
    lines.append("")

    # Transformer spot-check (E2)
    sample_rows = [r for r in rows if r["pred_V_transformer"]]
    if sample_rows:
        match_count = sum(
            1 for r in sample_rows
            if str(r["transformer_matches_dsl"]).lower() == "true"
        )
        lines.append("## Transformer ≡ DSL (E2 sample)")
        lines.append("")
        lines.append(
            f"- Sampled circuits: {len(sample_rows)}, "
            f"transformer_matches_dsl=True: {match_count}"
        )
        lines.append("")

    # Heat diffusion (E11)
    e11_csv = os.path.join(RESULTS_DIR, "heat_diffusion_E11.csv")
    if os.path.exists(e11_csv):
        with open(e11_csv, encoding="utf-8") as f:
            e11 = list(csv.DictReader(f))
        lines.append("## Heat diffusion (E11)")
        lines.append("")
        lines.append("Pass tolerance: `err_vs_jacobi <= tol_V` (V_STEP-compounded "
                     "quantization floor on multi-node Jacobi).")
        lines.append("")
        lines.append("| Grid | N | T | Pred | Jacobi ref | Analytical | err vs Jacobi | err vs Analytical | tol | Pass |")
        lines.append("|------|---|---|------|------------|------------|---------------|-------------------|-----|------|")
        for r in e11:
            lines.append(
                f"| {r.get('grid','')} | {r.get('N','')} | {r.get('T','')} | "
                f"{r.get('pred_V','')} | {r.get('jacobi_ref_V','')} | "
                f"{r.get('analytical_V','')} | "
                f"{r.get('err_vs_jacobi','')} | {r.get('err_vs_analytical','')} | "
                f"{r.get('tol_V','')} | {r.get('pass','')} |"
            )
        lines.append("")

    path = os.path.join(RESULTS_DIR, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
