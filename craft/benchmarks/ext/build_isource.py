"""Build per-circuit transformer weights using the I-source-aware DSL graph.

Mirrors `build.build_for_circuit` but uses CircuitMachineI / RBSORCircuitMachineI
in place of the standard ones. Produces `model_<ID>_<algorithm>_isrc.bin` files
to keep them separate from the existing models.

Usage:
    python -m benchmarks.ext.build_isource --benchmark pde IEEE_14_001 --algorithm jacobi
    python -m benchmarks.ext.build_isource --benchmark ieee IEEE_14_001 --algorithm rbsor
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BENCHMARKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_BENCHMARKS_DIR, "results")
MODEL_DIR = _BENCHMARKS_DIR  # store models alongside benchmarks/
PDE_DATASET = os.path.join(RESULTS_DIR, "pde_dataset.jsonl")
IEEE_DATASET = os.path.join(RESULTS_DIR, "ieee_dataset.jsonl")


def load_problem(pid: str, dataset_path: str) -> dict:
    with open(dataset_path) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == pid:
                return c
    raise KeyError(f"Problem {pid} not found in {dataset_path}")


def build_for_problem(
    pid: str,
    dataset_path: str,
    T: int | None,
    out_path: str | None,
    plan_only: bool,
    algorithm: str = "jacobi",
    v_step: int | None = None,
    k_levels: int | None = None,
    omega_cap: float | None = None,
) -> dict:
    """Build a transformer for one PDE/IEEE problem. Mirrors build.build_for_circuit."""
    from transformer_vm.model.weights import build_model, save_weights
    from transformer_vm.scheduler.milp import milp_schedule

    from benchmarks.ext.parse_isource import parse_netlist_isource

    c = load_problem(pid, dataset_path)
    netlist = c["Netlist"]
    target = int(c["Target_Node"])

    pc = parse_netlist_isource(netlist)

    if algorithm == "jacobi":
        from benchmarks.ext.interpreter_isource import CircuitMachineI
        from jacobi_reference import _auto_T
        if T is None:
            T = _auto_T(pc.num_nodes)
        log.info("Problem %s [jacobi+i]: N=%d, target=%d, T=%d, has_I=%s",
                 pid, pc.num_nodes, target, T,
                 any(abs(x) > 1e-12 for x in pc.i_inj_norm))
        machine = CircuitMachineI(pc, target, T, v_step=v_step, k_levels=k_levels)
    elif algorithm == "rbsor":
        from coloring import two_color
        from experiments.spectrum import compute_rho_kappa
        from rbsor_reference import auto_T_rbsor, omega_opt

        from benchmarks.ext.interpreter_rbsor_isource import RBSORCircuitMachineI
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        if omega_cap is not None:
            omega = min(omega, float(omega_cap))
        if T is None:
            T = auto_T_rbsor(pc.num_nodes, omega)
        red_order, black_order, conflicts = two_color(pc)
        log.info("Problem %s [rbsor+i]: N=%d, target=%d, T=%d, "
                 "rho=%.4f omega=%.4f red=%d black=%d conflicts=%d",
                 pid, pc.num_nodes, target, T, rho, omega,
                 len(red_order), len(black_order), len(conflicts))
        machine = RBSORCircuitMachineI(
            pc, target, T, omega, red_order, black_order,
            v_step=v_step, k_levels=k_levels,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    t0 = time.time()
    pg, meta = machine.build()
    log.info("Graph built in %.2fs: %d dims, %d lookups",
             time.time() - t0, len(pg.all_dims), len(pg.all_lookups))

    if plan_only:
        t1 = time.time()
        sched = milp_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
        log.info("MILP done in %.2fs: n_layers=%d, d_model=%d",
                 time.time() - t1, sched["num_layers"], sched["width"])
        return sched

    t1 = time.time()
    model, all_tokens, tok_to_idx_map, _ = build_model(program_graph=pg)
    log.info("Weights built in %.2fs", time.time() - t1)

    actual_d_model = model.tok.weight.shape[1]
    actual_n_layers = len(model.attn)
    actual_d_ffn = model.ff_in[0].weight.shape[0] // 2
    n_params = sum(p.numel() for p in model.parameters())

    if out_path is None:
        suffix = "_jacobi_isrc" if algorithm == "jacobi" else "_rbsor_isrc"
        out_path = os.path.join(MODEL_DIR, f"model_{pid}{suffix}.bin")

    save_weights(model, all_tokens, out_path)
    log.info("Saved: %s  (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    return {
        "problem": pid,
        "N": pc.num_nodes,
        "T": T,
        "n_layers": actual_n_layers,
        "d_model": actual_d_model,
        "d_ffn": actual_d_ffn,
        "vocab": len(all_tokens),
        "n_params": n_params,
        "model_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build per-problem I-source transformer")
    parser.add_argument("problem_id")
    parser.add_argument("--benchmark", choices=["pde", "ieee"], required=True)
    parser.add_argument("--algorithm", choices=["jacobi", "rbsor"], default="jacobi")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--v-step", type=int, default=None)
    parser.add_argument("--k-levels", type=int, default=None)
    parser.add_argument("--omega-cap", type=float, default=None)
    args = parser.parse_args()

    dataset_path = PDE_DATASET if args.benchmark == "pde" else IEEE_DATASET
    result = build_for_problem(
        args.problem_id, dataset_path, args.T, args.out, args.plan_only,
        args.algorithm, v_step=args.v_step, k_levels=args.k_levels,
        omega_cap=args.omega_cap,
    )
    if isinstance(result, dict) and "model_path" in result:
        print(f"\nBuild complete: {result}")


if __name__ == "__main__":
    main()
