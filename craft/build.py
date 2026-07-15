"""Build per-circuit transformer weights for a single circuit.

Usage:
    python build.py CKT_0001             # builds model_CKT_0001.bin
    python build.py CKT_0001 --T 200    # override iteration count
    python build.py CKT_0001 --out custom.bin
    python build.py CKT_0001 --plan-only # just run MILP, print stats

Output: model_<ID>.bin next to this script. MILP always writes plan.yaml here.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 — sys.path shim; must remain first

import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)


def load_circuit(cid: str, dataset_path: str | None = None) -> dict:
    # Read DATASET_PATH dynamically from the module so monkey-patches in
    # callers (e.g. heat_hull_bench injecting a synthetic dataset) take effect.
    if dataset_path is None:
        dataset_path = DATASET_PATH
    with open(dataset_path) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                return c
    raise KeyError(f"Circuit {cid} not found in {dataset_path}")


def build_for_circuit(
    cid: str,
    T: int | None,
    out_path: str | None,
    plan_only: bool,
    algorithm: str = "jacobi",
    v_step: int | None = None,
    k_levels: int | None = None,
    omega_cap: float | None = None,
):
    from parse import parse_netlist
    from transformer_vm.model.weights import build_model, save_weights
    from transformer_vm.scheduler.milp import milp_schedule

    c = load_circuit(cid)
    netlist = c["Netlist"]
    target = int(c["Target_Node"])

    pc = parse_netlist(netlist)

    if algorithm == "jacobi":
        from interpreter import CircuitMachine
        from jacobi_reference import _auto_T
        if T is None:
            T = _auto_T(pc.num_nodes)
        log.info("Circuit %s [jacobi]: N=%d, target=%d, T=%d, v_step=%s, k_levels=%s",
                 cid, pc.num_nodes, target, T, v_step, k_levels)
        machine = CircuitMachine(pc, target, T, v_step=v_step, k_levels=k_levels)
    elif algorithm == "rbsor":
        import sys as _sys
        _RBSOR = os.path.join(os.path.dirname(__file__), "rbsor")
        if _RBSOR not in _sys.path:
            _sys.path.insert(0, _RBSOR)
        from coloring import two_color
        from experiments.spectrum import compute_rho_kappa
        from rbsor_interpreter import RBSORCircuitMachine
        from rbsor_reference import auto_T_rbsor, omega_opt
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        if omega_cap is not None:
            omega = min(omega, float(omega_cap))
        if T is None:
            T = auto_T_rbsor(pc.num_nodes, omega)
        red_order, black_order, conflicts = two_color(pc)
        log.info("Circuit %s [rbsor]: N=%d, target=%d, T=%d, "
                 "rho=%.4f omega=%.4f cap=%s red=%d black=%d conflicts=%d v_step=%s",
                 cid, pc.num_nodes, target, T, rho, omega, omega_cap,
                 len(red_order), len(black_order), len(conflicts), v_step)
        machine = RBSORCircuitMachine(
            pc, target, T, omega, red_order, black_order,
            v_step=v_step, k_levels=k_levels,
        )
    elif algorithm == "cadj":
        import sys as _sys
        _CADJ = os.path.join(os.path.dirname(__file__), "cadj")
        if _CADJ not in _sys.path:
            _sys.path.insert(0, _CADJ)
        from cadj_interpreter import CADJCircuitMachine
        from cadj_reference import auto_T_cadj
        from experiments.spectrum import compute_eigenvalue_bounds
        lam_min, lam_max = compute_eigenvalue_bounds(pc)
        if T is None:
            T = auto_T_cadj(pc.num_nodes, lam_min, lam_max)
        log.info("Circuit %s [cadj]: N=%d, target=%d, T=%d, lam=[%.4f,%.4f] v_step=%s",
                 cid, pc.num_nodes, target, T, lam_min, lam_max, v_step)
        machine = CADJCircuitMachine(
            pc, target, T, lam_min, lam_max,
            v_step=v_step, k_levels=k_levels,
        )
    elif algorithm == "direct":
        import sys as _sys
        _CADJ = os.path.join(os.path.dirname(__file__), "cadj")
        if _CADJ not in _sys.path:
            _sys.path.insert(0, _CADJ)
        from direct_interpreter import DirectCircuitMachine
        log.info("Circuit %s [direct]: N=%d, target=%d, v_step=%s",
                 cid, pc.num_nodes, target, v_step)
        machine = DirectCircuitMachine(pc, target, v_step=v_step, k_levels=k_levels)
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    t0 = time.time()
    pg, meta = machine.build()
    log.info("Graph built in %.2fs: %d dims, %d lookups",
             time.time() - t0, len(pg.all_dims), len(pg.all_lookups))

    if plan_only:
        # Run MILP and return stats only — no weight build.
        t1 = time.time()
        sched = milp_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
        log.info("MILP done in %.2fs: n_layers=%d, d_model=%d",
                 time.time() - t1, sched["num_layers"], sched["width"])
        return sched

    # Build weights (milp_schedule runs internally, writes plan.yaml).
    t1 = time.time()
    model, all_tokens, tok_to_idx_map, _ = build_model(program_graph=pg)
    log.info("Weights built in %.2fs", time.time() - t1)

    actual_d_model = model.tok.weight.shape[1]
    actual_n_layers = len(model.attn)
    actual_d_ffn = model.ff_in[0].weight.shape[0] // 2
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: d_model=%d, n_layers=%d, d_ffn=%d, vocab=%d, params=%s",
             actual_d_model, actual_n_layers, actual_d_ffn, len(all_tokens),
             f"{n_params:,}")

    if out_path is None:
        suffix = "" if algorithm == "jacobi" else f"_{algorithm}"
        out_path = os.path.join(
            os.path.dirname(__file__), f"model_{cid}{suffix}.bin"
        )

    save_weights(model, all_tokens, out_path)
    log.info("Saved: %s  (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    return {
        "circuit": cid,
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
    parser = argparse.ArgumentParser(description="Build per-circuit Jacobi transformer")
    parser.add_argument("circuit_id", help="Circuit ID, e.g. CKT_0001")
    parser.add_argument("--T", type=int, default=None, help="Iteration count (default: auto)")
    parser.add_argument("--out", type=str, default=None, help="Output .bin path")
    parser.add_argument("--plan-only", action="store_true",
                        help="Only run MILP and print stats, no weight build")
    parser.add_argument("--algorithm", choices=["jacobi", "rbsor", "cadj", "direct"], default="jacobi",
                        help="Iterative solver to compile (default: jacobi)")
    parser.add_argument("--v-step", type=int, default=None,
                        help="Quantization step in scaled units (500 = 0.05V default)")
    parser.add_argument("--k-levels", type=int, default=None,
                        help="Number of v_k vocabulary tokens (480 default)")
    parser.add_argument("--omega-cap", type=float, default=None,
                        help="(rbsor only) clamp omega <= cap (e.g. 1.5)")
    args = parser.parse_args()

    result = build_for_circuit(
        args.circuit_id, args.T, args.out, args.plan_only, args.algorithm,
        v_step=args.v_step, k_levels=args.k_levels, omega_cap=args.omega_cap,
    )
    if isinstance(result, dict) and "model_path" in result:
        print(f"\nBuild complete: {result}")


if __name__ == "__main__":
    main()
