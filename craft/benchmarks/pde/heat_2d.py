"""2D heat-diffusion (Laplace, Dirichlet BC) → SPICE netlist + ground truth.

Reuses the existing experiments.heat_netlist.thermal_grid_to_spice() as the
generator (V+R only, no DSL change needed).

This file just enumerates parameter sweeps to produce a benchmark dataset.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401 — sys.path shim

from experiments.heat_netlist import thermal_grid_to_spice


def heat_problems() -> list[dict]:
    """Enumerate a fixed set of heat problems.

    Returns one dict per problem with keys: ID, Description, Netlist,
    Target_Node (str), Ground_Truth_Vout, Complexity, family, rows, cols.

    Each problem has the same JSONL schema as the existing 154-circuit dataset
    so the existing build/runner pipelines work unchanged.
    """
    problems: list[dict] = []
    pid = 0

    # 5x5 grid (N=26 nodes incl. GND): 5 BC variations
    # 7x7 grid (N=50): 4 BC variations
    # 10x10 grid (N=101): 3 BC variations  -- tests the Hull KV + RB-SOR scale
    grid_sizes = [
        (5, 5, [
            (10.0, 0.0, 0.0, 0.0),    # asymmetric, top-only hot
            (10.0, 5.0, 0.0, 0.0),    # 2-side
            (10.0, 0.0, 5.0, 0.0),
            (8.0,  2.0, 4.0, 6.0),    # all 4 different
            (12.0, 3.0, 0.0, 0.0),
        ]),
        (7, 7, [
            (10.0, 0.0, 0.0, 0.0),
            (10.0, 5.0, 0.0, 0.0),
            (8.0,  2.0, 4.0, 6.0),
            (15.0, 0.0, 0.0, 5.0),
        ]),
        (10, 10, [
            (10.0, 0.0, 0.0, 0.0),
            (8.0,  2.0, 4.0, 6.0),
            (15.0, 0.0, 0.0, 5.0),
        ]),
    ]

    for rows, cols, bc_list in grid_sizes:
        for bc in bc_list:
            top_t, bot_t, left_t, right_t = bc
            netlist, target_node, analytical = thermal_grid_to_spice(
                rows, cols,
                top_temp=top_t, bottom_temp=bot_t,
                left_temp=left_t, right_temp=right_t,
            )
            pid += 1
            problems.append({
                "ID": f"PDE_HEAT_{pid:04d}",
                "Description": f"Heat 2D {rows}x{cols} BC=({top_t},{bot_t},{left_t},{right_t})",
                "Netlist": netlist,
                "Target_Node": str(target_node),
                "Ground_Truth_Vout": float(analytical),
                "Complexity": "Hard" if rows * cols > 25 else "Intermediate",
                "family": "heat_2d",
                "rows": rows,
                "cols": cols,
                "boundary": list(bc),
            })

    return problems


if __name__ == "__main__":
    probs = heat_problems()
    print(f"Generated {len(probs)} heat problems.")
    for p in probs[:3]:
        print(f"  {p['ID']}  {p['Description']}  truth={p['Ground_Truth_Vout']:.4f}V  target={p['Target_Node']}")
        print(f"    (netlist starts with: {p['Netlist'].splitlines()[0]})")
