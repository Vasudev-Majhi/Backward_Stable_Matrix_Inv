"""2D Poisson equation (-∇²u = f, Dirichlet BC) → SPICE netlist with I-sources.

Discretization (5-point stencil, unit grid spacing, conductance = 1):
    Σ_neighbors (u_n - u_nbr) = f_n      for interior nodes n
i.e. sum of conductances · u_n - Σ conductances · u_nbr = f_n

In the resistor-network analogy:
    - Each interior node has degree 4 (or less on boundary), unit resistors
    - Boundary nodes are Dirichlet (V-sources to GND)
    - Forcing term f_n is implemented as a CURRENT SOURCE injecting f_n amperes
      into node n (since I = (Σ G) · V_n - Σ G · V_nbr matches the discretized
      Poisson equation when conductances are 1)

This file uses the I-source extension (ext/parse_isource.py) — these netlists
will not parse with the standard parse.py.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

import numpy as np


def _grid_to_spice_node(i: int, j: int, cols: int) -> int:
    """grid[i,j] -> SPICE node id (1-indexed; 0 is GND)."""
    return i * cols + j + 1


def poisson_grid_to_spice(
    rows: int,
    cols: int,
    top_bc: float,
    bottom_bc: float,
    left_bc: float,
    right_bc: float,
    forcing: callable | float = 0.0,
) -> tuple[str, int, float]:
    """Generate a SPICE netlist for the 2D Poisson Dirichlet problem.

    forcing: either a float (constant forcing) or a callable f(i, j) -> float
             returning the forcing at grid cell (i, j).

    Returns (netlist_text, target_node_0indexed, analytical_solution).
    The target is the center-most interior cell.

    NOTE: the netlist contains `I<name> N+ 0 <amperes>` lines that are only
    parsed by `benchmarks.ext.parse_isource`. The standard parse.py will
    silently skip them.
    """
    if rows < 3 or cols < 3:
        raise ValueError("grid must be at least 3x3")

    if not callable(forcing):
        f_val = float(forcing)
        forcing = lambda i, j, _f=f_val: _f  # noqa: E731

    R = 1.0  # unit resistor
    lines: list[str] = [f"* Poisson 2D {rows}x{cols} BC=({top_bc},{bottom_bc},{left_bc},{right_bc})"]

    def boundary_temp(i: int, j: int) -> float | None:
        if i == 0:
            return top_bc
        if i == rows - 1:
            return bottom_bc
        if j == 0:
            return left_bc
        if j == cols - 1:
            return right_bc
        return None

    # V-sources for boundary nodes
    v_count = 0
    for i in range(rows):
        for j in range(cols):
            t = boundary_temp(i, j)
            if t is None:
                continue
            node = _grid_to_spice_node(i, j, cols)
            v_count += 1
            lines.append(f"V{v_count} {node} 0 {t:.4f}")

    # Resistors (4-neighbor stencil)
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

    # Current sources for forcing term at interior nodes
    i_count = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            f_ij = forcing(i, j)
            if f_ij == 0.0:
                continue
            node = _grid_to_spice_node(i, j, cols)
            i_count += 1
            lines.append(f"I{i_count} {node} 0 {f_ij:.4f}")

    lines.append(".op")
    lines.append(".end")

    ci, cj = rows // 2, cols // 2
    target_node = _grid_to_spice_node(ci, cj, cols)

    analytical = _analytical_solve(
        rows, cols, top_bc, bottom_bc, left_bc, right_bc, forcing, target_node
    )

    return "\n".join(lines), target_node, analytical


def _analytical_solve(
    rows, cols, top_bc, bottom_bc, left_bc, right_bc, forcing, target_spice_node,
) -> float:
    """Direct solve of the discretized Poisson Dirichlet system.

    KCL at interior node (i,j) with unit conductances:
        4 u_ij - u_(i-1,j) - u_(i+1,j) - u_(i,j-1) - u_(i,j+1) = f_ij
    Boundary nodes are fixed by Dirichlet BC.
    """
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
            if i == 0:
                is_fixed[k] = True
                fixed_v[k] = top_bc
            elif i == rows - 1:
                is_fixed[k] = True
                fixed_v[k] = bottom_bc
            elif j == 0:
                is_fixed[k] = True
                fixed_v[k] = left_bc
            elif j == cols - 1:
                is_fixed[k] = True
                fixed_v[k] = right_bc

    for i in range(rows):
        for j in range(cols):
            k = idx(i, j)
            if is_fixed[k]:
                continue
            # Σ G·u_n - Σ G·u_nbr = f_n  → 4 u_n - sum nbr = f_n
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                nk = idx(ni, nj)
                A[k, k] += 1.0
                if is_fixed[nk]:
                    b[k] += fixed_v[nk]
                else:
                    A[k, nk] -= 1.0
            b[k] += forcing(i, j)

    free = [k for k in range(total) if not is_fixed[k]]
    free_map = {k: ii for ii, k in enumerate(free)}
    idx_arr = np.array(free)
    A_ff = A[np.ix_(idx_arr, idx_arr)]
    b_f = b[idx_arr]
    v_free = np.linalg.solve(A_ff, b_f)

    target_grid = target_spice_node - 1
    if is_fixed[target_grid]:
        return float(fixed_v[target_grid])
    return float(v_free[free_map[target_grid]])


def poisson_problems() -> list[dict]:
    """Enumerate Poisson problems for the benchmark."""
    problems: list[dict] = []
    pid = 0

    # Forcing patterns: const, point source at center, gaussian
    def const_f(val):
        return lambda i, j, _v=val: _v

    def gaussian_f(amp, sigma, ci, cj):
        return lambda i, j, _a=amp, _s=sigma, _ci=ci, _cj=cj: (
            _a * float(np.exp(-((i - _ci) ** 2 + (j - _cj) ** 2) / (2 * _s ** 2)))
        )

    grid_sizes = [
        (5, 5),
        (7, 7),
        (10, 10),
    ]

    for rows, cols in grid_sizes:
        ci, cj = rows // 2, cols // 2
        cases = [
            (("const_small",  "Constant forcing 0.5"), 10.0, 0.0, 0.0, 0.0, const_f(0.5)),
            (("const_zero",   "No forcing (Laplace)"), 10.0, 0.0, 0.0, 0.0, const_f(0.0)),
            (("gaussian",     "Gaussian source at center, amp=1.0"), 5.0, 5.0, 5.0, 5.0,
             gaussian_f(1.0, max(rows, cols) / 4.0, ci, cj)),
        ]
        for (key, desc), top, bot, left, right, fn in cases:
            netlist, target_node, analytical = poisson_grid_to_spice(
                rows, cols, top, bot, left, right, fn,
            )
            pid += 1
            problems.append({
                "ID": f"PDE_POIS_{pid:04d}",
                "Description": f"Poisson 2D {rows}x{cols} {desc}",
                "Netlist": netlist,
                "Target_Node": str(target_node),
                "Ground_Truth_Vout": float(analytical),
                "Complexity": "Hard" if rows * cols > 25 else "Intermediate",
                "family": "poisson_2d",
                "rows": rows,
                "cols": cols,
                "forcing_kind": key,
                "boundary": [top, bot, left, right],
            })

    return problems


if __name__ == "__main__":
    probs = poisson_problems()
    print(f"Generated {len(probs)} Poisson problems.")
    for p in probs[:3]:
        print(f"  {p['ID']}  {p['Description']}  truth={p['Ground_Truth_Vout']:.4f}V")
