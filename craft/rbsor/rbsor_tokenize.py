"""SPICE netlist -> Forward RB-SOR token sequence.

Layout (stride 2 throughout, exactly 2N tokens per iteration block — same
total block size as Jacobi, but nodes are reordered as red ++ black):

  pos 0                              : start
  pos 2*i+1, 2*i+2  for i in 0..N-1  : (skip, v_init(node))   where node = (red ++ black)[i]
  pos 2N + 1 + 2(t*N + i)            : update token at color_idx i in iteration t
                                          i in 0..N_R-1   -> up_red_<n>_p{t%2}   (n = red_order[i])
                                          i in N_R..N-1   -> up_blk_<n>_p{t%2}   (n = black_order[i-N_R])
  pos 2N + 2 + 2(t*N + i)            : <PRED>
  pos 2(T+1)*N + 1                   : readout
  pos 2(T+1)*N + 2                   : <PRED>
  pos 2(T+1)*N + 3                   : halt

CRITICAL: the init block must be in red ++ black order so that the t=0 update
token's `v_old` fetch (offset = -2N) lands on its own node's v_init. The
DSL graph in rbsor_interpreter.py assumes this layout exactly.
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

from coloring import color_idx_map, two_color
from parse import ParsedCircuit, parse_netlist
from rbsor_reference import K_LEVELS, SCALE, V_STEP, auto_T_rbsor, omega_opt

PREDICTED = "<PRED>"


def _v_token_for(voltage_scaled_int: int, v_step: int = V_STEP, k_levels: int = K_LEVELS) -> str:
    k = round(voltage_scaled_int / v_step)
    k = max(0, min(k_levels - 1, k))
    return f"v_{k}"


def tokenize_rbsor(
    netlist: str,
    target_node: int,
    T: int | None = None,
    omega: float | None = None,
    red_order: list[int] | None = None,
    black_order: list[int] | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuit, int, float, list[int], list[int]]:
    """Return (tokens, pc, T_used, omega_used, red_order, black_order).

    If omega / red_order / black_order are None, they are auto-derived from the
    parsed circuit via spectrum.compute_rho_kappa + omega_opt + two_color.
    """
    vs = V_STEP if v_step is None else int(v_step)
    kl = K_LEVELS if k_levels is None else int(k_levels)

    pc = parse_netlist(netlist)

    if red_order is None or black_order is None:
        red_order, black_order, _ = two_color(pc)

    if omega is None:
        from experiments.spectrum import compute_rho_kappa
        rho, _ = compute_rho_kappa(pc)
        omega = omega_opt(rho)

    if T is None:
        T = auto_T_rbsor(pc.num_nodes, omega)

    cidx = color_idx_map(red_order, black_order)
    N = pc.num_nodes
    if N != len(red_order) + len(black_order):
        raise ValueError(
            f"coloring covers {len(red_order) + len(black_order)} nodes, expected {N}"
        )

    # Concatenated color order: position i in the iteration block <-> node id.
    block_order = list(red_order) + list(black_order)

    toks: list[str] = ["start"]

    # Init block in red ++ black order.
    for node in block_order:
        toks.append("skip")
        v0_scaled = int(round(pc.fixed_voltage[node] * SCALE))
        toks.append(_v_token_for(v0_scaled, vs, kl))

    # Iteration blocks.
    n_red = len(red_order)
    for t in range(T):
        p = t % 2
        for i, node in enumerate(block_order):
            prefix = "red" if i < n_red else "blk"
            toks.append(f"up_{prefix}_{node}_p{p}")
            toks.append(PREDICTED)

    # Readout + final prediction + halt.
    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    return toks, pc, T, omega, red_order, black_order


if __name__ == "__main__":
    import json
    import os
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    )
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            c = json.loads(line)
            toks, pc, T, omega, red, black = tokenize_rbsor(c["Netlist"], int(c["Target_Node"]))
            n_pred = sum(1 for t in toks if t == PREDICTED)
            print(f"{c['ID']} N={pc.num_nodes} T={T} omega={omega:.3f} "
                  f"|red|={len(red)} |black|={len(black)} "
                  f"total_tokens={len(toks)} predicted={n_pred}")
            print(f"  head: {toks[:14]}")
