"""Pure-Python Jacobi and RB-SOR reference solvers WITH current-source support.

Mirrors `jacobi_reference.jacobi_solve_parsed` and
`rbsor_reference.rbsor_solve_parsed` exactly, but adds the I_inj_norm term
in the same place the DSL does (after Σ w·v / SCALE for Jacobi; inside
v_jacobi for RB-SOR).
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

from rbsor_reference import K_LEVELS, SCALE, V_STEP, _quantize_to_vk

from benchmarks.ext.parse_isource import ParsedCircuitWithI


def jacobi_solve_parsed_isource(
    pc: ParsedCircuitWithI, target: int, T: int,
) -> float:
    """Plain Jacobi with I-source support. Mirrors jacobi_reference.jacobi_solve_parsed."""
    n = pc.num_nodes
    v = [int(round(pc.fixed_voltage[i] * SCALE)) if pc.is_fixed[i] else 0
         for i in range(n)]

    # i_inj_norm in scaled int volts
    i_inj_int = [int(round(pc.i_inj_norm[i] * SCALE)) for i in range(n)]

    for _ in range(T):
        v_new = [0] * n
        for node in range(n):
            if pc.is_fixed[node]:
                v_new[node] = int(round(pc.fixed_voltage[node] * SCALE))
                continue
            acc = 0
            for nb, w in pc.out_edges_norm[node]:
                acc += w * v[nb]
            # acc // SCALE matches jacobi_reference.py (integer floor).
            # Add i_inj_int directly (already in scaled volts).
            v_new[node] = acc // SCALE + i_inj_int[node]
        v = v_new

    return v[target] / SCALE


def rbsor_solve_parsed_isource(
    pc: ParsedCircuitWithI,
    target: int,
    T: int,
    omega: float,
    red_order: list[int],
    black_order: list[int],
    v_step: int = V_STEP,
    k_levels: int = K_LEVELS,
) -> float:
    """RB-SOR with I-source support. Mirrors rbsor_reference.rbsor_solve_parsed."""
    n = pc.num_nodes
    v = [_quantize_to_vk(int(round(pc.fixed_voltage[i] * SCALE)), v_step, k_levels)
         if pc.is_fixed[i]
         else _quantize_to_vk(0, v_step, k_levels)
         for i in range(n)]

    # i_inj_norm in (true float) scaled-volts; do the float divide like the DSL.
    i_inj_scaled = [pc.i_inj_norm[i] * SCALE for i in range(n)]

    red_set = set(red_order)

    def update(node: int, v_prev_node: int, v_for_neighbors: list[int]) -> int:
        if pc.is_fixed[node]:
            return _quantize_to_vk(
                int(round(pc.fixed_voltage[node] * SCALE)), v_step, k_levels
            )
        acc = 0
        for nb, w in pc.out_edges_norm[node]:
            acc += w * v_for_neighbors[nb]
        # Float divide to match the DSL semantics exactly (see comment block in
        # rbsor_reference.py:139-145).
        v_jacobi = acc / SCALE + i_inj_scaled[node]
        sor = (1.0 - omega) * v_prev_node + omega * v_jacobi
        return _quantize_to_vk(sor, v_step, k_levels)

    for _ in range(T):
        v_prev = list(v)
        for node in red_order:
            v[node] = update(node, v_prev[node], v_prev)
        v_for_blk = [v[i] if i in red_set else v_prev[i] for i in range(n)]
        for node in black_order:
            v[node] = update(node, v_prev[node], v_for_blk)

    return v[target] / SCALE


def numpy_direct_solve(pc: ParsedCircuitWithI, target: int) -> float:
    """Direct linear solve via numpy LU. The 'gold standard' truth.

    Solves the nodal admittance system  G · v = b  where:
      - G is the Laplacian-like conductance matrix (using out_edges_norm)
      - b includes both the fixed-voltage contributions and I_inj_norm

    Note: out_edges_norm is the *normalized* conductance (sum to SCALE per row),
    not raw. We use raw resistors here for numerical stability.
    """
    import numpy as np

    n = pc.num_nodes
    free = [i for i in range(n) if not pc.is_fixed[i]]
    free_idx = {nd: k for k, nd in enumerate(free)}

    if not free:
        return float(pc.fixed_voltage[target])

    A = np.zeros((len(free), len(free)), dtype=np.float64)
    rhs = np.zeros(len(free), dtype=np.float64)

    # Build per-node total conductance and adjacency from raw resistors.
    sum_g = [0.0] * n
    adj_g: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        adj_g[a].append((b, g))
        adj_g[b].append((a, g))
        sum_g[a] += g
        sum_g[b] += g

    for nd in free:
        row = free_idx[nd]
        A[row, row] = sum_g[nd]
        for nb, g in adj_g[nd]:
            if pc.is_fixed[nb]:
                rhs[row] += g * pc.fixed_voltage[nb]
            elif nb in free_idx:
                A[row, free_idx[nb]] -= g
        # Add I-source injection on the RHS (current INTO node n).
        # KCL: sum_out_currents = I_inj  =>  Σ G·(V_n - V_nbr) = I_inj
        # => G·V_n - Σ G·V_nbr = I_inj. So I_inj_raw goes on the RHS.
        # But we have only i_inj_norm = I_inj_raw / sum_g. So I_inj_raw = i_inj_norm * sum_g.
        rhs[row] += pc.i_inj_norm[nd] * sum_g[nd]

    if pc.is_fixed[target]:
        return float(pc.fixed_voltage[target])

    v_free = np.linalg.solve(A, rhs)
    return float(v_free[free_idx[target]])
