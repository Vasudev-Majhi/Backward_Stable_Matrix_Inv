"""Tokenizers for the I-source extension.

The tokenization itself does NOT depend on I-sources (the token sequence layout
is the same V+R-only one — start, skip, v_k, up_n_p, readout, halt). I-sources
only affect the EMBEDDING values via the new `i_inj_slot` InputDimension.

These wrappers parse with `parse_netlist_isource` so callers get back a
`ParsedCircuitWithI`, then delegate to the existing `tokenize_netlist.tokenize`
or `rbsor_tokenize.tokenize_rbsor` for the actual token sequence.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

from benchmarks.ext.parse_isource import ParsedCircuitWithI, parse_netlist_isource
from interpreter import K_LEVELS, V_STEP


def tokenize_isource(
    netlist: str,
    target_node: int,
    T: int | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuitWithI, int]:
    """Drop-in replacement for tokenize_netlist.tokenize that returns
    ParsedCircuitWithI in place of ParsedCircuit. Token sequence is identical."""
    from tokenize_netlist import _auto_T, _v_token_for, PREDICTED  # type: ignore

    vs = V_STEP if v_step is None else int(v_step)
    kl = K_LEVELS if k_levels is None else int(k_levels)
    SCALE = 10000

    pc = parse_netlist_isource(netlist)
    if T is None:
        T = _auto_T(pc.num_nodes)

    toks: list[str] = ["start"]
    for n in range(pc.num_nodes):
        toks.append("skip")
        v0_scaled = int(round(pc.fixed_voltage[n] * SCALE))
        toks.append(_v_token_for(v0_scaled, vs, kl))

    for t in range(T):
        p = t % 2
        for n in range(pc.num_nodes):
            toks.append(f"up_{n}_p{p}")
            toks.append(PREDICTED)

    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    return toks, pc, T


def tokenize_rbsor_isource(
    netlist: str,
    target_node: int,
    T: int | None = None,
    omega: float | None = None,
    red_order: list[int] | None = None,
    black_order: list[int] | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuitWithI, int, float, list[int], list[int]]:
    """Drop-in replacement for rbsor_tokenize.tokenize_rbsor that returns
    ParsedCircuitWithI."""
    from coloring import color_idx_map, two_color
    from rbsor_reference import K_LEVELS as RB_K, SCALE as RB_S, V_STEP as RB_V
    from rbsor_reference import auto_T_rbsor, omega_opt
    from rbsor_tokenize import PREDICTED, _v_token_for as _vk

    vs = RB_V if v_step is None else int(v_step)
    kl = RB_K if k_levels is None else int(k_levels)

    pc = parse_netlist_isource(netlist)

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

    block_order = list(red_order) + list(black_order)

    toks: list[str] = ["start"]
    for node in block_order:
        toks.append("skip")
        v0_scaled = int(round(pc.fixed_voltage[node] * RB_S))
        toks.append(_vk(v0_scaled, vs, kl))

    n_red = len(red_order)
    for t in range(T):
        p = t % 2
        for i, node in enumerate(block_order):
            prefix = "red" if i < n_red else "blk"
            toks.append(f"up_{prefix}_{node}_p{p}")
            toks.append(PREDICTED)

    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    return toks, pc, T, omega, red_order, black_order
