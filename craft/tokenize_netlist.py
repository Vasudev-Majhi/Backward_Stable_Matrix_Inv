"""SPICE netlist -> transformer token sequence (per-circuit Jacobi).

Produces the FIXED portion of the sequence. Predicted positions (after each
update token and the readout token) are filled in at runtime by the runner
from model outputs.

Layout (stride 2 throughout):
  pos 0              : start
  pos 2n+1, 2n+2     : (skip, v_{k_init(n)})  for n in 0..N-1   (iter -1 = init)
  pos 2N+1+2(tN+n)   : update_{n}_p{t%2}      for t in 0..T-1, n in 0..N-1
  pos 2N+2+2(tN+n)   : <predicted v-token>    (runner fills in)
  pos 2(T+1)N+1      : readout
  pos 2(T+1)N+2      : <predicted final v>    (runner fills in)
  pos 2(T+1)N+3      : halt
"""
from __future__ import annotations

from parse import ParsedCircuit, parse_netlist
from interpreter import V_STEP, K_LEVELS, SCALE

PREDICTED = "<PRED>"   # placeholder string; runner substitutes model argmax


def _v_token_for(voltage_scaled_int: int, v_step: int = V_STEP, k_levels: int = K_LEVELS) -> str:
    k = round(voltage_scaled_int / v_step)
    k = max(0, min(k_levels - 1, k))
    return f"v_{k}"


def _auto_T(n_nodes: int) -> int:
    return max(1000, 50 * n_nodes)


def tokenize(
    netlist: str,
    target_node: int,
    T: int | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuit, int]:
    """Return (token_list_with_placeholders, parsed, T_used).

    Optional quantization overrides v_step / k_levels must match what the
    interpreter's DSL graph uses (CircuitMachine with same args).
    """
    vs = V_STEP if v_step is None else int(v_step)
    kl = K_LEVELS if k_levels is None else int(k_levels)

    pc = parse_netlist(netlist)
    if T is None:
        T = _auto_T(pc.num_nodes)

    toks: list[str] = ["start"]

    # Init block: (skip, v_init(n)) pairs for n in 0..N-1
    for n in range(pc.num_nodes):
        toks.append("skip")
        v0_scaled = int(round(pc.fixed_voltage[n] * SCALE))
        toks.append(_v_token_for(v0_scaled, vs, kl))

    # Iteration blocks: (update_n_p, PRED) pairs
    for t in range(T):
        p = t % 2
        for n in range(pc.num_nodes):
            toks.append(f"up_{n}_p{p}")
            toks.append(PREDICTED)

    # Readout + final prediction + halt
    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    return toks, pc, T


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/circuit_dataset_rv.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            c = json.loads(line)
            toks, pc, T = tokenize(c["Netlist"], int(c["Target_Node"]))
            n_pred = sum(1 for t in toks if t == PREDICTED)
            print(f"{c['ID']} N={pc.num_nodes} T={T} total_tokens={len(toks)} predicted={n_pred}")
            # Peek at first 15 tokens
            print(f"  head: {toks[:15]}")
