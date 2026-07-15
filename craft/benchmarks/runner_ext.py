"""Runner for I-source-aware compiled-transformer inference.

Mirrors `runner.run_circuit` but:
  - Reads problems from `pde_dataset.jsonl` or `ieee_dataset.jsonl`
    (instead of `circuit_dataset_rv.jsonl`).
  - Uses `parse_netlist_isource` + `tokenize_isource`/`tokenize_rbsor_isource`
    so the I-source embedding values flow into the model.
  - Loads `model_<ID>_<algo>_isrc.bin` (built by `build_isource.py`).

Cache class is selected via `runner.CACHE_CLASS` (mirrors the existing
hull_kv/runner_hull.py pattern). Default: StandardKVCache. Set to
HullKVCache before calling run_problem() for the O(log n) speedup.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import json
import logging
import os
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PREDICTED = "<PRED>"

_BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_BENCHMARKS_DIR, "results")
MODEL_DIR = _BENCHMARKS_DIR


def load_problem(pid: str, dataset_path: str) -> dict:
    with open(dataset_path) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == pid:
                return c
    raise KeyError(f"Problem {pid} not in {dataset_path}")


def load_dataset(dataset_path: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with open(dataset_path) as f:
        for line in f:
            c = json.loads(line)
            by_id[c["ID"]] = c
    return by_id


def _model_path(pid: str, algorithm: str = "jacobi", model_suffix: str = "") -> str:
    suffix = "_jacobi_isrc" if algorithm == "jacobi" else "_rbsor_isrc"
    return os.path.join(MODEL_DIR, f"model_{pid}{suffix}{model_suffix}.bin")


def run_problem(
    pid: str,
    dataset_path: str,
    T: int | None,
    tol: float,
    verbose: bool,
    algorithm: str = "jacobi",
    v_step: int | None = None,
    k_levels: int | None = None,
    omega_cap: float | None = None,
    model_path_override: str | None = None,
    model_suffix: str = "",
) -> tuple[str, float, float, float, dict]:
    """Run inference for one problem. Returns (status, pred_v, truth_v, elapsed, info)."""
    import runner  # the existing module — we use its CACHE_CLASS + _forward_step
    from interpreter import V_STEP as DEFAULT_V_STEP
    from transformer_vm.model.weights import load_weights

    from benchmarks.ext.parse_isource import parse_netlist_isource

    effective_v_step = DEFAULT_V_STEP if v_step is None else int(v_step)

    problem = load_problem(pid, dataset_path)
    netlist = problem["Netlist"]
    target = int(problem["Target_Node"])
    truth = problem["Ground_Truth_Vout"]

    info: dict = {"problem_id": pid, "family": problem.get("family", "")}

    t0 = time.time()

    mpath = model_path_override if model_path_override else _model_path(pid, algorithm, model_suffix)
    if not os.path.exists(mpath):
        return "NO_MODEL", 0.0, truth, time.time() - t0, info

    model, all_tokens, tok_to_idx = load_weights(mpath)
    model.eval()

    pc = parse_netlist_isource(netlist)
    info["N"] = pc.num_nodes
    info["max_degree"] = max(pc.degree) if pc.degree else 0

    if algorithm == "jacobi":
        from jacobi_reference import _auto_T

        from benchmarks.ext.tokenize_isource import tokenize_isource
        if T is None:
            T = _auto_T(pc.num_nodes)
        fixed_toks, _, _ = tokenize_isource(
            netlist, target, T=T, v_step=v_step, k_levels=k_levels,
        )
    elif algorithm == "rbsor":
        from coloring import two_color
        from experiments.spectrum import compute_rho_kappa
        from rbsor_reference import auto_T_rbsor, omega_opt

        from benchmarks.ext.tokenize_isource import tokenize_rbsor_isource
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)
        if omega_cap is not None:
            omega = min(omega, float(omega_cap))
        red_order, black_order, _ = two_color(pc)
        if T is None:
            T = auto_T_rbsor(pc.num_nodes, omega)
        fixed_toks, _, T_used, omega_used, _, _ = tokenize_rbsor_isource(
            netlist, target, T=T, omega=omega,
            red_order=red_order, black_order=black_order,
            v_step=v_step, k_levels=k_levels,
        )
        T = T_used
        info["rho_J"] = rho
        info["omega"] = omega_used
        info["n_red"] = len(red_order)
        info["n_black"] = len(black_order)
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    info["T_used"] = T
    info["seq_length"] = len(fixed_toks)

    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = runner.CACHE_CLASS(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for layer_idx in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[layer_idx][h]:
                    cache.set_tiebreak(layer_idx, h, True)

    predicted_vs: list[str] = []
    pos = 0

    with torch.no_grad():
        for tok in fixed_toks:
            if tok == PREDICTED:
                logits = model.head(x)  # noqa: F821
                best_name = None
                best_score = -1e18
                for name, idx in tok_to_idx.items():
                    if name.startswith("v_"):
                        s = logits[idx].item()
                        if s > best_score:
                            best_score = s
                            best_name = name
                if best_name is None:
                    best_name = "v_0"
                predicted_vs.append(best_name)
                tok_idx = tok_to_idx[best_name]
                x = runner._forward_step(model, cache, tok_idx, pos)
                pos += 1
            else:
                if tok not in tok_to_idx:
                    log.warning("Unknown token %r for %s", tok, pid)
                    tok = "start"
                tok_idx = tok_to_idx[tok]
                x = runner._forward_step(model, cache, tok_idx, pos)
                pos += 1

    final_vk = predicted_vs[-1]
    assert final_vk.startswith("v_"), f"expected v_<k>, got {final_vk}"
    k = int(final_vk.split("_")[1])
    pred_v = k * effective_v_step / 10000.0

    elapsed = time.time() - t0
    err = abs(pred_v - truth)
    status = "PASS" if err <= tol else "FAIL"
    info["pred_v"] = pred_v
    info["truth_v"] = truth
    info["abs_error"] = err
    info["elapsed"] = elapsed
    return status, pred_v, truth, elapsed, info
