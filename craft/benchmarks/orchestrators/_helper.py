"""Shared helper for the four PDE/IEEE orchestrators.

Each orchestrator picks (benchmark, algorithm) ∈ {pde,ieee} × {jacobi,rbsor}
and then loops over its problem set. The build/run/append-CSV logic is shared
here so we don't duplicate ~400 lines four times.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import csv
import json
import logging
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

log = logging.getLogger(__name__)

_BENCHMARKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_BENCHMARKS_DIR, "results")
MODEL_DIR = _BENCHMARKS_DIR
PDE_DATASET = os.path.join(RESULTS_DIR, "pde_dataset.jsonl")
IEEE_DATASET = os.path.join(RESULTS_DIR, "ieee_dataset.jsonl")


FIELDS = [
    "problem_id", "family", "complexity", "N", "max_degree", "target_node",
    "rho_J", "omega", "T_used", "n_red", "n_black",
    "pred_V_dsl", "ref_V_solver", "ref_V_direct", "truth_V",
    "abs_error", "pass_fail", "dsl_matches_solver",
    "seq_length", "run_time_s", "n_layers", "d_model",
    "build_status", "run_status", "error",
]


def _setup_hull_cache(use_hull: bool) -> str:
    """Inject HullKVCache or StandardKVCache into runner.CACHE_CLASS."""
    import runner
    if use_hull:
        from transformer_vm.attention.hull_cache import HullKVCache
        HullKVCache(1, 1)  # JIT-compile the C++ extension
        runner.CACHE_CLASS = HullKVCache
        return "HullKVCache"
    else:
        from transformer_vm.attention.standard_cache import StandardKVCache
        runner.CACHE_CLASS = StandardKVCache
        return "StandardKVCache"


def _build_if_missing(
    pid: str, dataset_path: str, model_path: str, algorithm: str,
    v_step: int | None = None, k_levels: int | None = None,
) -> tuple[str, dict | None]:
    """Build the model.bin if missing. Returns (status, build_info)."""
    if os.path.exists(model_path):
        return "cached", None
    from benchmarks.ext.build_isource import build_for_problem
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull):
            info = build_for_problem(
                pid, dataset_path, T=None, out_path=model_path,
                plan_only=False, algorithm=algorithm,
                v_step=v_step, k_levels=k_levels,
            )
        return "built", info
    except Exception as e:
        log.warning(f"Build failed for {pid}: {e}\n{traceback.format_exc()}")
        return f"build_error: {e}", None
    finally:
        devnull.close()


def _solve_with_python_reference(
    netlist: str, target: int, algorithm: str, T_used: int,
) -> tuple[float, float, dict]:
    """Run the Python reference (Jacobi or RB-SOR) for ground truth.

    Returns (ref_V_solver, ref_V_direct, extra_info).
    """
    from benchmarks.ext.parse_isource import parse_netlist_isource
    from benchmarks.ext.reference_isource import (
        jacobi_solve_parsed_isource,
        numpy_direct_solve,
        rbsor_solve_parsed_isource,
    )

    pc = parse_netlist_isource(netlist)
    extra: dict = {}

    if algorithm == "jacobi":
        ref_solver = jacobi_solve_parsed_isource(pc, target, T_used)
    elif algorithm == "rbsor":
        from coloring import two_color
        from experiments.spectrum import compute_rho_kappa
        from rbsor_reference import omega_opt
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        red_order, black_order, _ = two_color(pc)
        ref_solver = rbsor_solve_parsed_isource(
            pc, target, T_used, omega, red_order, black_order,
        )
        extra["rho_J"] = rho
        extra["omega"] = omega
        extra["n_red"] = len(red_order)
        extra["n_black"] = len(black_order)
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    ref_direct = numpy_direct_solve(pc, target)
    return ref_solver, ref_direct, extra


def run_one(
    pid: str, problem: dict, dataset_path: str, algorithm: str,
    tol: float, force_rebuild: bool,
    v_step: int | None = None, k_levels: int | None = None,
    model_suffix: str = "",
) -> dict:
    """Build (if missing) + run + write a CSV row dict for one problem."""
    row = {fn: "" for fn in FIELDS}
    row["problem_id"] = pid
    row["family"] = problem.get("family", "")
    row["complexity"] = problem.get("Complexity", "")
    row["truth_V"] = problem["Ground_Truth_Vout"]

    try:
        netlist = problem["Netlist"]
        target = int(problem["Target_Node"])

        # ── Build model ─────────────────────────────────────────────
        algo_suffix = "_jacobi_isrc" if algorithm == "jacobi" else "_rbsor_isrc"
        model_path = os.path.join(MODEL_DIR, f"model_{pid}{algo_suffix}{model_suffix}.bin")
        if force_rebuild and os.path.exists(model_path):
            os.remove(model_path)
        bstatus, binfo = _build_if_missing(
            pid, dataset_path, model_path, algorithm,
            v_step=v_step, k_levels=k_levels,
        )
        row["build_status"] = bstatus
        if binfo:
            row["n_layers"] = binfo["n_layers"]
            row["d_model"] = binfo["d_model"]
            row["N"] = binfo["N"]
            row["target_node"] = target
            row["T_used"] = binfo["T"]
        elif bstatus == "cached":
            try:
                from transformer_vm.model.weights import load_weights
                m, _, _ = load_weights(model_path)
                row["n_layers"] = len(m.attn)
                row["d_model"] = m.tok.weight.shape[1]
            except Exception:
                pass

        if bstatus.startswith("build_error"):
            row["run_status"] = "skipped"
            row["error"] = bstatus
            return row

        # ── Run model ─────────────────────────────────────────────
        from benchmarks.runner_ext import run_problem
        try:
            status, pred_v, truth, elapsed, info = run_problem(
                pid, dataset_path, T=None, tol=tol, verbose=False,
                algorithm=algorithm,
                v_step=v_step, k_levels=k_levels,
                model_suffix=model_suffix,
            )
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"] = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"] = "PASS" if status == "PASS" else "FAIL"
            row["run_status"] = status
            row["run_time_s"] = f"{elapsed:.2f}"
            row["seq_length"] = info.get("seq_length", "")
            row["N"] = info.get("N", row.get("N", ""))
            row["max_degree"] = info.get("max_degree", "")
            row["T_used"] = info.get("T_used", row.get("T_used", ""))
            if "rho_J" in info:
                row["rho_J"] = f"{info['rho_J']:.6f}"
            if "omega" in info:
                row["omega"] = f"{info['omega']:.4f}"
            if "n_red" in info:
                row["n_red"] = info["n_red"]
            if "n_black" in info:
                row["n_black"] = info["n_black"]

            # ── Python reference + direct solve ─────────────────────
            try:
                T_used = int(info.get("T_used") or row.get("T_used") or 0)
                ref_solver, ref_direct, extra = _solve_with_python_reference(
                    netlist, target, algorithm, T_used,
                )
                row["ref_V_solver"] = f"{ref_solver:.4f}"
                row["ref_V_direct"] = f"{ref_direct:.4f}"
                row["dsl_matches_solver"] = (
                    "Y" if abs(pred_v - ref_solver) < 0.01 else "N"
                )
            except Exception as e:
                log.warning(f"{pid}: reference failed: {e}")
        except Exception as e:
            row["run_status"] = "run_error"
            row["error"] = repr(e)
            log.warning(f"{pid}: run failed: {e}\n{traceback.format_exc()}")

    except Exception as e:
        row["error"] = repr(e)
        log.error(f"{pid}: unexpected: {e}\n{traceback.format_exc()}")

    return row


def write_rows(out_csv: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def orchestrate(
    benchmark: str, algorithm: str, out_csv: str,
    use_hull: bool = True, limit: int | None = None,
    smoke: bool = False, tol: float = 0.05, force_rebuild: bool = False,
    v_step: int | None = None, k_levels: int | None = None,
    model_suffix: str = "",
) -> None:
    """Run benchmark × algorithm sweep and write incremental CSV checkpoints."""
    if benchmark == "pde":
        dataset_path = PDE_DATASET
    elif benchmark == "ieee":
        dataset_path = IEEE_DATASET
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found. Run "
            f"`python benchmarks/{benchmark}/make_{benchmark}_dataset.py` first."
        )

    cache_name = _setup_hull_cache(use_hull=use_hull)
    print(
        f"[orchestrator] benchmark={benchmark} algorithm={algorithm} "
        f"cache={cache_name} v_step={v_step} k_levels={k_levels} suffix={model_suffix!r}",
        flush=True,
    )

    problems: list[dict] = []
    with open(dataset_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if smoke:
        problems = problems[:1]
    elif limit is not None:
        problems = problems[:limit]

    print(f"[orchestrator] {len(problems)} problems", flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0

    for i, p in enumerate(problems):
        pid = p["ID"]
        t0 = time.time()
        row = run_one(
            pid, p, dataset_path, algorithm, tol, force_rebuild,
            v_step=v_step, k_levels=k_levels, model_suffix=model_suffix,
        )
        rows.append(row)
        write_rows(out_csv, rows)

        bucket = "?"
        if row["pass_fail"] == "PASS":
            pass_n += 1
            bucket = "PASS"
        elif row["pass_fail"] == "FAIL":
            fail_n += 1
            bucket = "FAIL"
        else:
            err_n += 1
            bucket = row["run_status"] or "ERR"

        dt = time.time() - t0
        print(
            f"[{i+1:>3}/{len(problems)}] {pid:<22} "
            f"N={row['N']:<3} T={row['T_used']:<5} "
            f"pred={row['pred_V_dsl']:>8} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>7} [{bucket}] {dt:.1f}s",
            flush=True,
        )

    print(
        f"\n[orchestrator] DONE  benchmark={benchmark}  algorithm={algorithm}  "
        f"pass={pass_n}  fail={fail_n}  err={err_n}  total={len(problems)}  "
        f"out={out_csv}",
        flush=True,
    )
