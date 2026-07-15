"""Run the merged LU+direct transformer on a single circuit.

The runner threads runtime values between stages via Hull KV-cache V-patching.
No Python arithmetic is performed -- it only reads values out of the hidden
state at known slot indices and writes them back into the cache.

Per-step patch contract (Hull cache only; StandardKVCache fallback also works):
    rhs_i  -> patch slot_x_value at pos_rhs_i with b_value (read from x[slot_b_value]).
    fwd_i  -> patch slot_x_value at pos_fwd_i with y_i = x[slot_x_new]; remember y_buf[i].
    bck_i  -> inject y_buf[i] into slot_y_input BEFORE the forward pass;
              patch slot_x_value at pos_bck_i with x_i so future bck_j and the
              readout's free-target fetch (clear_key=1-is_bck_tok) see x_i.
    readout -> no patch; the quadratic v_k score determines the prediction.
"""
from __future__ import annotations

import _path  # noqa: F401

import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

from cadj_reference import K_LEVELS, SCALE, V_STEP  # type: ignore
from parse import parse_netlist  # type: ignore

from lu_direct_tokenize import PREDICTED, tokenize_lu_direct


def _select_cache_class(use_hull: bool):
    if use_hull:
        try:
            from transformer_vm.attention.hull_cache import HullKVCache  # type: ignore
            return HullKVCache, True
        except Exception as e:
            log.warning("Hull cache unavailable (%s); falling back to StandardKVCache", e)
    from transformer_vm.attention.standard_cache import StandardKVCache  # type: ignore
    return StandardKVCache, False


def _forward_step(model, cache, x_initial: torch.Tensor) -> torch.Tensor:
    """One transformer pass given a pre-built x_initial (already pos-encoded)."""
    x = x_initial.clone()
    for li, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
        out = cache.layer_step(li, k, q, v)
        x = x + attn.out_proj(out)
        gate, val = ff_in(x).chunk(2, dim=-1)
        x = x + ff_out(F.relu(gate) * val)
    return x


def _build_x(model, tok_idx: int, pos: int) -> torch.Tensor:
    from transformer_vm.model.transformer import add_position_encoding  # type: ignore
    x = model.tok.weight[tok_idx].clone()
    add_position_encoding(x, pos)
    return x


def _vcache_patch(
    model, cache, x_init: torch.Tensor, n_layers: int,
    primary_value: float, primary_slots: list[int],
) -> None:
    """Re-insert (K, V) at the most-recent position with primary_value placed in
    every primary_slot. Setting multiple slots to the same value is defence against
    MILP slot colocation -- if x_value_slot and x_new happen to share a slot in a
    given layer, both writes still produce the right value (LU project's runner_lu
    n=6 slot-collision fix). Order matters: x_new MUST be set last so it overrides
    a colliding x_value write.
    """
    x_corrected = x_init.clone()
    for slot in primary_slots:
        x_corrected[slot] = float(primary_value)
    for li in range(n_layers):
        new_kqv = model.attn[li].in_proj_weight @ x_corrected
        new_k, _, new_v = new_kqv.chunk(3, dim=-1)
        if hasattr(cache, "_vals"):                  # StandardKVCache
            cache._vals[li][-1] = new_v.clone()
        elif hasattr(cache, "insert_v"):              # HullKVCache
            cache.insert_v(li, new_k, new_v)


def run_one(
    model_path: str,
    netlist: str,
    truth: float,
    tol: float,
    v_step: int | None,
    k_levels: int | None,
    verbose: bool,
    use_hull: bool,
) -> tuple[str, float, float, float]:
    """Run the merged transformer once. Returns (status, pred_v, truth_v, elapsed)."""
    from transformer_vm.model.weights import load_weights  # type: ignore

    effective_v_step = V_STEP if v_step is None else int(v_step)

    t0 = time.time()

    sidecar_path = model_path + ".slots.json"
    if not os.path.exists(model_path):
        return "NO_MODEL", 0.0, truth, time.time() - t0
    if not os.path.exists(sidecar_path):
        return "NO_SIDECAR", 0.0, truth, time.time() - t0

    with open(sidecar_path) as f:
        side = json.load(f)
    slot_x_value = int(side["slot_x_value"])
    slot_y_input = int(side["slot_y_input"])
    slot_b_value = int(side["slot_b_value"])
    slot_x_new   = int(side["slot_x_new"])
    slot_v_score_source = int(side.get("slot_v_score_source", -1))
    slot_emit_v_gate    = int(side.get("slot_emit_v_gate", -1))
    n_free       = int(side["n_free"])

    model, all_tokens, tok_to_idx = load_weights(model_path)
    model.eval()

    tokens, _, _ = tokenize_lu_direct(netlist, v_step=v_step, k_levels=k_levels)

    cache_cls, _using_hull = _select_cache_class(use_hull)
    n_layers = len(model.attn)
    n_heads = model.attn[0].num_heads
    cache = cache_cls(n_layers, n_heads)
    if hasattr(model, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
        for li in range(n_layers):
            for h in range(n_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)

    y_buf = [0.0] * max(n_free, 1)
    x_buf = [0.0] * max(n_free, 1)
    pos = 0
    pred_v = 0.0

    with torch.no_grad():
        for tok in tokens:
            if tok == PREDICTED:
                logits = model.head(x)  # type: ignore[reportPossiblyUnbound]
                best_name = "v_0"
                best_score = -1e30
                v_scores = []
                for name, idx in tok_to_idx.items():
                    if name.startswith("v_"):
                        s = logits[idx].item()
                        v_scores.append((s, name, int(name.split("_")[1])))
                        if s > best_score:
                            best_score = s
                            best_name = name
                k_pred = int(best_name.split("_")[1])
                pred_v = k_pred * effective_v_step / SCALE
                if verbose:
                    log.info("  pos %d PRED -> %s (%.4fV)", pos, best_name, pred_v)
                    v_scores.sort(reverse=True)
                    for sc, nm, k_idx in v_scores[:5]:
                        log.info("    top: %s score=%.3e (vk=%d)", nm, sc, k_idx * effective_v_step)
                    log.info("    @readout: x_new=%.3e b_value=%.3e v_score=%.3e emit=%.3e",
                             x[slot_x_new].item(), x[slot_b_value].item(),
                             x[slot_v_score_source].item() if slot_v_score_source >= 0 else float("nan"),
                             x[slot_emit_v_gate].item() if slot_emit_v_gate >= 0 else float("nan"))
                tok_idx = tok_to_idx[best_name]
                x = _forward_step(model, cache, _build_x(model, tok_idx, pos))
                pos += 1
                continue

            if tok not in tok_to_idx:
                tok = "start"
            tok_idx = tok_to_idx[tok]
            x_init = _build_x(model, tok_idx, pos)

            # bck_i needs runtime y_input injection BEFORE the forward pass.
            if tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_init[slot_y_input] = x_init[slot_y_input] + float(y_buf[i])

            x = _forward_step(model, cache, x_init)

            if tok.startswith("rhs_"):
                i = int(tok.split("_", 1)[1])
                b_i = float(x[slot_b_value].item())
                if verbose:
                    log.info("  pos %d %s  b=%.6e", pos, tok, b_i)
                _vcache_patch(
                    model, cache, x_init, n_layers,
                    primary_value=b_i,
                    primary_slots=[slot_x_value, slot_b_value, slot_x_new],
                )

            elif tok.startswith("fwd_"):
                i = int(tok.split("_", 1)[1])
                y_i = float(x[slot_x_new].item())
                y_buf[i] = y_i
                if verbose:
                    log.info("  pos %d %s  y=%.6e", pos, tok, y_i)
                _vcache_patch(
                    model, cache, x_init, n_layers,
                    primary_value=y_i,
                    primary_slots=[slot_x_value, slot_x_new],
                )

            elif tok.startswith("bck_"):
                i = int(tok.split("_", 1)[1])
                x_i = float(x[slot_x_new].item())
                x_buf[i] = x_i
                if verbose:
                    log.info("  pos %d %s  x=%.6e", pos, tok, x_i)
                _vcache_patch(
                    model, cache, x_init, n_layers,
                    primary_value=x_i,
                    primary_slots=[slot_x_value, slot_x_new],
                )
            pos += 1

    elapsed = time.time() - t0
    err = abs(pred_v - truth)
    status = "PASS" if err <= tol else "FAIL"
    return status, pred_v, truth, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("circuit_id")
    p.add_argument("--dataset", default=os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    ))
    p.add_argument("--model-dir", default=os.path.dirname(__file__))
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--v-step", type=int, default=None)
    p.add_argument("--k-levels", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-hull", action="store_true",
                   help="(default) Use StandardKVCache. Hull's hard-attention is "
                        "incompatible with our V-cache patching for some circuits.")
    p.add_argument("--hull", action="store_true",
                   help="Force HullKVCache (experimental; some circuits fail).")
    args = p.parse_args()

    with open(args.dataset) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == args.circuit_id:
                break
        else:
            raise SystemExit(f"circuit {args.circuit_id} not found")

    model_path = os.path.join(args.model_dir, f"model_{args.circuit_id}_lu_direct.bin")
    status, pred_v, truth_v, elapsed = run_one(
        model_path=model_path,
        netlist=c["Netlist"],
        truth=float(c["Ground_Truth_Vout"]),
        tol=args.tol,
        v_step=args.v_step,
        k_levels=args.k_levels,
        verbose=args.verbose,
        use_hull=args.hull,
    )
    err = abs(pred_v - truth_v)
    print(f"{status} {args.circuit_id} pred={pred_v:.4f} truth={truth_v:.4f} "
          f"err={err:.4f} time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
