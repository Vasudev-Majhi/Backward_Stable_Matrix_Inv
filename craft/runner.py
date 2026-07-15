"""Phase 5: Run per-circuit Jacobi transformer on the dataset.

Loads model_<ID>.bin (built by build.py), feeds the interleaved token sequence
through the real PyTorch transformer, reads the predicted v_k tokens, and
compares the final voltage against Ground_Truth_Vout.

Usage:
    python runner.py CKT_0001                  # single circuit
    python runner.py CKT_0001,CKT_0023         # comma-separated list
    python runner.py 10                        # first 10 circuits
    python runner.py all                       # all circuits

Flags:
    --T <int>         override iteration count
    --tol <float>     pass tolerance in volts (default 0.05)
    --model-dir <dir> directory containing model_*.bin files
    --verbose         print per-token prediction trace
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 — sys.path shim

import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "CRAFT_DATASET",
    "dataset/circuit_dataset_rv.jsonl",
)
MODEL_DIR = os.path.dirname(__file__)
PREDICTED = "<PRED>"

# Cache class is module-level so wrappers (e.g. hull_kv/runner_hull.py) can swap
# in HullKVCache without duplicating runner code. Default = O(n) softmax cache.
from transformer_vm.attention.standard_cache import StandardKVCache as _DefaultCache
CACHE_CLASS = _DefaultCache


def load_dataset(path: str | None = None) -> dict[str, dict]:
    if path is None:
        path = DATASET_PATH
    by_id: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            by_id[c["ID"]] = c
    return by_id


def _model_path(cid: str, model_dir: str, algorithm: str = "jacobi") -> str:
    suffix = "" if algorithm == "jacobi" else f"_{algorithm}"
    return os.path.join(model_dir, f"model_{cid}{suffix}.bin")


def _forward_step(model, cache, tok_idx: int, pos: int) -> torch.Tensor:
    """Run one token through the transformer, return final hidden state x."""
    x = model.tok.weight[tok_idx].clone()
    # Positional encoding — same as add_position_encoding in transformer.py
    from transformer_vm.model.transformer import add_position_encoding
    add_position_encoding(x, pos)

    for layer_idx, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
        out = cache.layer_step(layer_idx, k, q, v)
        x = x + attn.out_proj(out)

        gate, val = ff_in(x).chunk(2, dim=-1)
        x = x + ff_out(F.relu(gate) * val)

    return x


def run_circuit(
    cid: str | None,
    model_dir: str,
    T: int | None,
    tol: float,
    verbose: bool,
    algorithm: str = "jacobi",
    v_step: int | None = None,
    k_levels: int | None = None,
    omega_cap: float | None = None,
    model_path_override: str | None = None,
    netlist_override: str | None = None,
    target_override: int | None = None,
    truth_override: float | None = None,
) -> tuple[str, float, float, float]:
    """Run one circuit. Returns (status, pred_v, truth_v, elapsed).

    If netlist_override/target_override/truth_override are given, skip dataset lookup.
    """
    from interpreter import V_STEP as DEFAULT_V_STEP
    from parse import parse_netlist
    from transformer_vm.model.weights import load_weights
    effective_v_step = DEFAULT_V_STEP if v_step is None else int(v_step)

    # ── Load dataset entry ──────────────────────────────────────────
    if netlist_override is not None:
        netlist = netlist_override
        target  = target_override
        truth   = truth_override
    else:
        by_id = load_dataset()
        c = by_id[cid]
        netlist = c["Netlist"]
        target = int(c["Target_Node"])
        truth = c["Ground_Truth_Vout"]

    t0 = time.time()

    # ── Load model ──────────────────────────────────────────────────
    mpath = model_path_override if model_path_override else _model_path(cid, model_dir, algorithm)
    if not os.path.exists(mpath):
        return "NO_MODEL", 0.0, truth, time.time() - t0

    model, all_tokens, tok_to_idx = load_weights(mpath)
    model.eval()

    # ── Build token sequence ─────────────────────────────────────────
    pc = parse_netlist(netlist)

    if algorithm == "jacobi":
        from jacobi_reference import _auto_T
        from tokenize_netlist import tokenize
        if T is None:
            T = _auto_T(pc.num_nodes)
        fixed_toks, _, _ = tokenize(netlist, target, T=T,
                                    v_step=v_step, k_levels=k_levels)
    elif algorithm == "cadj":
        import sys as _sys
        _CADJ = os.path.join(os.path.dirname(__file__), "cadj")
        if _CADJ not in _sys.path:
            _sys.path.insert(0, _CADJ)
        from cadj_tokenize import tokenize_cadj
        from cadj_reference import auto_T_cadj
        from experiments.spectrum import compute_eigenvalue_bounds
        lam_min, lam_max = compute_eigenvalue_bounds(pc)
        if T is None:
            T = auto_T_cadj(pc.num_nodes, lam_min, lam_max)
        fixed_toks, _, T_used, _, _ = tokenize_cadj(
            netlist, target, T=T, lam_min=lam_min, lam_max=lam_max,
            v_step=v_step, k_levels=k_levels,
        )
        T = T_used
        if verbose:
            log.info("CADJ T=%d lam=[%.4f,%.4f]", T, lam_min, lam_max)
    elif algorithm == "rbsor":
        import sys as _sys
        _RBSOR = os.path.join(os.path.dirname(__file__), "rbsor")
        if _RBSOR not in _sys.path:
            _sys.path.insert(0, _RBSOR)
        from coloring import two_color
        from rbsor_tokenize import tokenize_rbsor
        from rbsor_reference import omega_opt as _opt, auto_T_rbsor as _auto
        from experiments.spectrum import compute_rho_kappa
        # Derive omega here (so we can apply the cap) instead of letting
        # tokenize_rbsor auto-derive it. Build site uses the same logic.
        rho, _ = compute_rho_kappa(pc)
        omega = _opt(rho)
        if omega_cap is not None:
            omega = min(omega, float(omega_cap))
        red_order, black_order, _ = two_color(pc)
        if T is None:
            T = _auto(pc.num_nodes, omega)
        fixed_toks, _, T_used, omega_used, _, _ = tokenize_rbsor(
            netlist, target, T=T, omega=omega,
            red_order=red_order, black_order=black_order,
            v_step=v_step, k_levels=k_levels,
        )
        T = T_used
        if verbose:
            log.info("RB-SOR T=%d omega=%.4f", T, omega_used)
    elif algorithm == "direct":
        import sys as _sys
        _CADJ = os.path.join(os.path.dirname(__file__), "cadj")
        if _CADJ not in _sys.path:
            _sys.path.insert(0, _CADJ)
        from direct_tokenize import tokenize_direct
        fixed_toks, _ = tokenize_direct(netlist, v_step=v_step, k_levels=k_levels)
        if verbose:
            log.info("direct: %d tokens (no iterations)", len(fixed_toks))
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    # ── Step-by-step interleaved inference ──────────────────────────
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = CACHE_CLASS(n_layers, n_heads)
    # Apply per-head tiebreak from the saved model. No-op for caches that
    # don't expose set_tiebreak (e.g. StandardKVCache uses softmax). Mirrors
    # transformer_vm.model.transformer.VanillaTransformer.generate_with_cache.
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
                # Score all tokens using the hidden state from the previous step.
                # (x was computed at the previous position — we saved it.)
                logits = model.head(x)  # type: ignore[reportPossiblyUnbound]
                # Pick the best v_k token (highest logit among v_ tokens).
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
                if verbose:
                    log.info("  pos %d PRED -> %s (score %.1f)", pos, best_name, best_score)

                # Feed the predicted token back.
                tok_idx = tok_to_idx[best_name]
                x = _forward_step(model, cache, tok_idx, pos)
                pos += 1
            else:
                if tok not in tok_to_idx:
                    # Token not in model vocab (shouldn't happen for valid circuits).
                    log.warning("Unknown token %r for %s", tok, cid)
                    tok = "start"
                tok_idx = tok_to_idx[tok]
                x = _forward_step(model, cache, tok_idx, pos)
                pos += 1

    # ── Parse final voltage ──────────────────────────────────────────
    final_vk = predicted_vs[-1]
    assert final_vk.startswith("v_"), f"expected v_<k>, got {final_vk}"
    k = int(final_vk.split("_")[1])
    pred_v = k * effective_v_step / 10000.0

    elapsed = time.time() - t0
    err = abs(pred_v - truth)
    status = "PASS" if err <= tol else "FAIL"
    return status, pred_v, truth, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run per-circuit Jacobi transformer")
    parser.add_argument("circuits", help="Circuit ID(s), comma list, count, or 'all'")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--tol", type=float, default=0.05, help="Pass tolerance in volts")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory with model_*.bin files")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--algorithm", choices=["jacobi", "rbsor", "cadj", "direct"], default="jacobi",
                        help="Solver to run (must match the build's --algorithm)")
    parser.add_argument("--v-step", type=int, default=None,
                        help="Quantization step in scaled units (must match build)")
    parser.add_argument("--k-levels", type=int, default=None,
                        help="Number of v_k tokens (must match build)")
    parser.add_argument("--omega-cap", type=float, default=None,
                        help="(rbsor only) clamp omega <= cap (must match build)")
    args = parser.parse_args()

    by_id = load_dataset()
    all_ids = list(by_id.keys())

    if args.circuits == "all":
        ids = all_ids
    elif "CKT_" in args.circuits:
        ids = args.circuits.split(",")
    else:
        limit = int(args.circuits)
        ids = all_ids[:limit]

    passed = failed = no_model = errors = 0
    for cid in ids:
        try:
            status, pred_v, truth, elapsed = run_circuit(
                cid, args.model_dir, args.T, args.tol, args.verbose,
                args.algorithm,
                v_step=args.v_step, k_levels=args.k_levels,
                omega_cap=args.omega_cap,
            )
            err = abs(pred_v - truth)
            if status == "NO_MODEL":
                no_model += 1
                print(f"SKIP  {cid:<10} no model.bin found")
            elif status == "PASS":
                passed += 1
                print(f"PASS  {cid:<10} pred={pred_v:.4f} truth={truth:.4f} "
                      f"err={err:.4f} time={elapsed:.1f}s")
            else:
                failed += 1
                print(f"FAIL  {cid:<10} pred={pred_v:.4f} truth={truth:.4f} "
                      f"err={err:.4f} time={elapsed:.1f}s")
        except Exception as e:
            errors += 1
            print(f"ERROR {cid}: {e}")

    total = len(ids)
    print(f"\n{passed}/{total} PASS  |  {failed} FAIL  |  "
          f"{no_model} SKIP  |  {errors} ERROR  (tol={args.tol}V)")


if __name__ == "__main__":
    main()
