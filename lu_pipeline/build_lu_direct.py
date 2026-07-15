"""Build per-circuit weights for the merged LU+direct transformer.

Outputs:
  model_<ID>_lu_direct.bin              -- transformer weights
  model_<ID>_lu_direct.bin.slots.json   -- sidecar with slot indices and layout

Build-time math is restricted to sparse-conductance accumulation and Doolittle
LU on A_FF -- no numpy.linalg.solve, ever.
"""
from __future__ import annotations

import _path  # noqa: F401

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
    out_path: str | None = None,
    plan_only: bool = False,
    v_step: int | None = None,
    k_levels: int | None = None,
):
    from parse import parse_netlist  # type: ignore
    from transformer_vm.model.weights import build_model, save_weights  # type: ignore
    from transformer_vm.scheduler.milp import milp_schedule  # type: ignore

    from lu_direct_interpreter import LuDirectCircuitMachine

    c = load_circuit(cid)
    # Cap MILP solve at 60s; HiGHS' proof-of-optimality phase doesn't move
    # d_model meaningfully but eats most of the build-time budget.
    os.environ.setdefault("MILP_TIME_LIMIT", "60")
    netlist = c["Netlist"]
    target = int(c["Target_Node"])
    pc = parse_netlist(netlist)

    log.info("Circuit %s [lu_direct]: N=%d, target=%d", cid, pc.num_nodes, target)
    machine = LuDirectCircuitMachine(pc, target, v_step=v_step, k_levels=k_levels)

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
    model, all_tokens, tok_to_idx_map, shared = build_model(program_graph=pg)
    log.info("Weights built in %.2fs", time.time() - t1)

    slot_of = shared[6]

    # Resolve slot indices for runner.
    def _slot(dim) -> int:
        if dim in slot_of:
            return int(slot_of[dim])
        # Fall back to type/name lookup for persisted dims (the user-facing
        # handle and the scheduled dim can be different objects).
        from transformer_vm.graph.core import PersistDimension  # type: ignore
        for d, s in slot_of.items():
            if isinstance(d, PersistDimension) and getattr(d, "name", None) == getattr(dim, "name", None):
                return int(s)
        raise RuntimeError(f"Could not locate slot for dim {dim!r}")

    x_value_dim = meta["x_value_slot_dim"]
    v_value_dim = meta["v_value_slot_dim"]
    y_input_dim = meta["y_input_slot_dim"]
    is_v_em_dim = meta["is_v_emission_dim"]
    b_value_dim = next(iter(meta["b_value_expr"].terms))
    x_new_dim   = next(iter(meta["x_new_expr"].terms))
    v_score_dim = next(iter(meta["v_score_source_expr"].terms))
    emit_dim    = next(iter(meta["emit_v_gate_expr"].terms))

    slot_x_value       = _slot(x_value_dim)
    slot_v_value       = _slot(v_value_dim)
    slot_y_input       = _slot(y_input_dim)
    slot_is_v_emission = _slot(is_v_em_dim)
    slot_b_value       = _slot(b_value_dim)
    slot_x_new         = _slot(x_new_dim)
    slot_v_score       = _slot(v_score_dim)
    slot_emit          = _slot(emit_dim)

    actual_d_model  = model.tok.weight.shape[1]
    actual_n_layers = len(model.attn)
    actual_d_ffn    = model.ff_in[0].weight.shape[0] // 2
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: d_model=%d, n_layers=%d, d_ffn=%d, vocab=%d, params=%s",
             actual_d_model, actual_n_layers, actual_d_ffn, len(all_tokens),
             f"{n_params:,}")
    log.info("Slots: x_value=%d v_value=%d y_input=%d is_v_em=%d b_value=%d x_new=%d",
             slot_x_value, slot_v_value, slot_y_input, slot_is_v_emission,
             slot_b_value, slot_x_new)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__),
                                f"model_{cid}_lu_direct.bin")
    save_weights(model, all_tokens, out_path)
    log.info("Saved: %s  (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    sidecar = {
        "circuit_id": cid,
        "algorithm": "lu_direct",
        "N": int(meta["N"]),
        "n_free": int(meta["n_free"]),
        "n_fixed": int(meta["n_fixed"]),
        "target_node": int(meta["target_node"]),
        "target_is_fixed": bool(meta["target_is_fixed"]),
        "target_free_index": int(meta["target_free_index"]),
        "free": list(meta["free"]),
        "fixed": list(meta["fixed"]),
        "pos_rhs_base": int(meta["pos_rhs_base"]),
        "pos_fwd_base": int(meta["pos_fwd_base"]),
        "pos_bck_base": int(meta["pos_bck_base"]),
        "pos_readout":  int(meta["pos_readout"]),
        "slot_x_value": slot_x_value,
        "slot_v_value": slot_v_value,
        "slot_y_input": slot_y_input,
        "slot_is_v_emission": slot_is_v_emission,
        "slot_b_value": slot_b_value,
        "slot_x_new":   slot_x_new,
        "slot_v_score_source": slot_v_score,
        "slot_emit_v_gate":    slot_emit,
    }
    with open(out_path + ".slots.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    log.info("Saved sidecar: %s.slots.json", out_path)

    return {
        "circuit": cid,
        "model_path": out_path,
        "n_layers": actual_n_layers,
        "d_model": actual_d_model,
        "n_free": int(meta["n_free"]),
        "n_fixed": int(meta["n_fixed"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit_id")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    args = ap.parse_args()

    result = build_for_circuit(
        args.circuit_id, args.out, args.plan_only,
        v_step=args.v_step, k_levels=args.k_levels,
    )
    if isinstance(result, dict) and "model_path" in result:
        print(f"\nBuild complete: {result}")


if __name__ == "__main__":
    main()
