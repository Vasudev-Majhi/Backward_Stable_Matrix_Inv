"""E11: generate SPICE netlists from 2D heat-diffusion grid specs.

A discrete 2D Laplacian with constant conductivity is identical to a uniform
resistor network: temperature ↔ voltage, thermal conductance ↔ electrical
conductance, fixed boundary temperature ↔ voltage source.

Layout:
  rows × cols grid of nodes. Boundary nodes (row 0, row rows-1, col 0,
  col cols-1) are fixed to user-specified temperatures. Interior nodes are
  free. Each pair of adjacent grid nodes is connected by a unit resistor.

Node numbering (SPICE 0-indexed, where 0 = GND):
  grid[i, j] maps to SPICE node n = i * cols + j + 1
  GND (SPICE node 0) is used for the V-source references only.

Target node: center-most interior cell.
Analytical solution: solve the full Laplacian system via numpy directly.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np


def _grid_to_spice_node(i: int, j: int, cols: int) -> int:
    """grid[i,j] -> SPICE node id (1-indexed). 0 is reserved for GND."""
    return i * cols + j + 1


def thermal_grid_to_spice(
    rows: int,
    cols: int,
    top_temp: float,
    bottom_temp: float,
    left_temp: float,
    right_temp: float,
    conductivity: float = 1.0,
) -> tuple[str, int, float]:
    """Generate a SPICE netlist for a thermal-grid Dirichlet problem.

    Returns (netlist_text, target_node_0indexed, analytical_solution).
    The target is the center-most interior node (rows//2, cols//2).

    NOTE: Use asymmetric boundary temperatures for E11. Symmetric BCs (e.g.
    top=10, others=0) tend to produce analytical answers that fall on exact
    quantization half-steps (multiples of V_STEP/2 = 0.025V), where the DSL's
    banker's rounding in the argmax over v_k tokens can get stuck in a biased
    fixed point. Asymmetric BCs avoid this numerical edge case.
    """
    if rows < 3 or cols < 3:
        raise ValueError("grid must be at least 3x3 (need at least one interior node)")

    R = 1.0 / conductivity  # ohms
    lines: list[str] = [f"* Thermal grid {rows}x{cols}"]

    # Identify boundary nodes -> unique SPICE fixed voltages.
    # We group boundary nodes by their side so we use 4 voltage sources at most
    # (top/bottom/left/right). But edges have corners assigned to specific sides:
    # For uniqueness, each boundary node gets ONE fixed voltage. We prefer this
    # corner assignment: top row takes precedence over left/right; bottom row
    # likewise; then left/right get the middle rows.
    def boundary_temp(i: int, j: int) -> float | None:
        if i == 0:
            return top_temp
        if i == rows - 1:
            return bottom_temp
        if j == 0:
            return left_temp
        if j == cols - 1:
            return right_temp
        return None  # interior

    # V-sources: one per unique boundary temperature.
    # SPICE requires each V-source to connect a unique net to GND. We emit
    # a separate V-source for each boundary node so parser handles them
    # straightforwardly — even if many share the same temperature.
    v_count = 0
    for i in range(rows):
        for j in range(cols):
            t = boundary_temp(i, j)
            if t is None:
                continue
            node = _grid_to_spice_node(i, j, cols)
            v_count += 1
            lines.append(f"V{v_count} {node} 0 {t:.4f}")

    # Resistors between each orthogonal neighbor pair (4-neighbor stencil).
    r_count = 0
    for i in range(rows):
        for j in range(cols):
            na = _grid_to_spice_node(i, j, cols)
            if j + 1 < cols:
                nb = _grid_to_spice_node(i, j + 1, cols)
                r_count += 1
                lines.append(f"R{r_count} {na} {nb} {R:.4f}")
            if i + 1 < rows:
                nb = _grid_to_spice_node(i + 1, j, cols)
                r_count += 1
                lines.append(f"R{r_count} {na} {nb} {R:.4f}")

    lines.append(".op")
    lines.append(".end")

    # Target = center interior node, index in the pc's 0-indexed numbering
    # (same as SPICE node). Note: parse.py will re-map contiguously; since all
    # our nodes are contiguous 0..N-1 already (0 = GND, 1..rows*cols = grid),
    # the mapping is identity.
    ci, cj = rows // 2, cols // 2
    target_spice_node = _grid_to_spice_node(ci, cj, cols)
    target_0indexed = target_spice_node  # parse.py keeps node IDs as integers

    # Analytical solution via direct linear solve.
    analytical = _analytical_solve(
        rows, cols, top_temp, bottom_temp, left_temp, right_temp, target_spice_node
    )

    netlist = "\n".join(lines)
    return netlist, target_0indexed, analytical


def _analytical_solve(
    rows: int,
    cols: int,
    top_temp: float,
    bottom_temp: float,
    left_temp: float,
    right_temp: float,
    target_spice_node: int,
) -> float:
    """Direct dense linear solve for the temperature at target_spice_node.

    Uses a 4-neighbor stencil with unit conductances. Boundary nodes are
    fixed via Dirichlet conditions.
    """
    # We don't need GND for the solve since V-sources are just Dirichlet
    # constraints on boundary grid nodes; solve the interior system directly.

    # Map (i,j) -> grid-local index (0..rows*cols - 1).
    def idx(i, j):
        return i * cols + j

    total = rows * cols
    A = np.zeros((total, total), dtype=np.float64)
    b = np.zeros(total)
    is_fixed = np.zeros(total, dtype=bool)
    fixed_v = np.zeros(total)

    for i in range(rows):
        for j in range(cols):
            k = idx(i, j)
            # Determine if boundary.
            if i == 0:
                is_fixed[k] = True
                fixed_v[k] = top_temp
                continue
            if i == rows - 1:
                is_fixed[k] = True
                fixed_v[k] = bottom_temp
                continue
            if j == 0:
                is_fixed[k] = True
                fixed_v[k] = left_temp
                continue
            if j == cols - 1:
                is_fixed[k] = True
                fixed_v[k] = right_temp
                continue

    # Build A matrix for interior nodes only.
    for i in range(rows):
        for j in range(cols):
            k = idx(i, j)
            if is_fixed[k]:
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                nk = idx(ni, nj)
                A[k, k] += 1.0  # conductance = 1
                if is_fixed[nk]:
                    b[k] += fixed_v[nk]
                else:
                    A[k, nk] -= 1.0

    free = [k for k in range(total) if not is_fixed[k]]
    free_map = {k: i for i, k in enumerate(free)}
    idx_arr = np.array(free)
    A_ff = A[np.ix_(idx_arr, idx_arr)]
    b_f = b[idx_arr]
    v_free = np.linalg.solve(A_ff, b_f)

    target_grid = target_spice_node - 1  # SPICE is 1-indexed; grid is 0-indexed
    if is_fixed[target_grid]:
        return float(fixed_v[target_grid])
    return float(v_free[free_map[target_grid]])


if __name__ == "__main__":
    import sys

    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    nl, target, ana = thermal_grid_to_spice(
        rows, cols, top_temp=10.0, bottom_temp=0.0, left_temp=0.0, right_temp=0.0
    )
    print(f"# {rows}x{cols} grid, target node = {target}, analytical = {ana:.4f}V")
    print(nl)
    print()

    # Parse with existing parser
    from parse import parse_netlist
    pc = parse_netlist(nl)
    print(f"Parsed: N={pc.num_nodes}, {len(pc.resistors)} resistors, "
          f"{sum(pc.is_fixed)} fixed nodes")
