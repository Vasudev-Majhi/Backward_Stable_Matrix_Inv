"""Bulk E1/E2/E3 runner: interleaved inference via the DSL evaluator.

Uses `fast_attention.make_fast_runtime` (O(1) per fetch). Produces output
that is bit-identical to what the compiled transformer would produce
(both compute exact hardmax attention over the same token sequence).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import time

from fast_attention import make_fast_runtime
from interpreter import CircuitMachine, V_STEP
from parse import ParsedCircuit, parse_netlist
from tokenize_netlist import PREDICTED, tokenize


def run_one_dsl(
    netlist: str,
    target_node: int,
    T: int,
    *,
    parsed: ParsedCircuit | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> dict:
    """Run the circuit transformer via the DSL evaluator.

    Optional `v_step` and `k_levels` override the default quantization
    (V_STEP=500 / K_LEVELS=480). Both the DSL graph AND the tokenizer must
    use matching values.

    Returns a dict with keys:
      pred_V, final_vk, predicted_tokens (list of v_k names in emission order),
      seq_length, runtime_s, v_step (the step actually used).
    """
    pc = parsed if parsed is not None else parse_netlist(netlist)

    vs = V_STEP if v_step is None else int(v_step)

    t0 = time.time()
    machine = CircuitMachine(pc, target_node, T, v_step=v_step, k_levels=k_levels)
    pg, meta = machine.build()
    rt = make_fast_runtime(pg)

    fixed_toks, _, _ = tokenize(netlist, target_node, T=T,
                                v_step=v_step, k_levels=k_levels)
    predicted_log: list[str] = []
    last_vals = None

    i = 0
    while i < len(fixed_toks):
        tok = fixed_toks[i]
        if tok == PREDICTED:
            predicted = rt.predict_next(last_vals)
            predicted_log.append(predicted)
            last_vals = rt.step(predicted)
        else:
            last_vals = rt.step(tok)
        i += 1

    rt.destroy()
    runtime_s = time.time() - t0

    final_vk = predicted_log[-1]
    if not final_vk.startswith("v_"):
        return {
            "pred_V": 0.0,
            "final_vk": final_vk,
            "predicted_tokens": predicted_log,
            "seq_length": len(fixed_toks),
            "runtime_s": runtime_s,
            "error": f"expected v_<k>, got {final_vk}",
        }

    k = int(final_vk.split("_")[1])
    pred_V = k * vs / 10000.0

    return {
        "pred_V": pred_V,
        "final_vk": final_vk,
        "v_step": vs,
        "predicted_tokens": predicted_log,
        "seq_length": len(fixed_toks),
        "runtime_s": runtime_s,
    }


if __name__ == "__main__":
    import json
    import sys

    path = "dataset/circuit_dataset_rv.jsonl"
    cid = sys.argv[1] if len(sys.argv) > 1 else "CKT_0001"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    with open(path) as f:
        for line in f:
            c = json.loads(line)
            if c["ID"] == cid:
                break
        else:
            raise SystemExit(f"{cid} not found")

    r = run_one_dsl(c["Netlist"], int(c["Target_Node"]), T)
    err = abs(r["pred_V"] - c["Ground_Truth_Vout"])
    status = "PASS" if err <= 0.05 else "FAIL"
    print(
        f"{status} {cid} T={T} pred={r['pred_V']:.4f}V "
        f"truth={c['Ground_Truth_Vout']:.4f}V err={err:.4f}V "
        f"seq={r['seq_length']} time={r['runtime_s']:.2f}s"
    )
