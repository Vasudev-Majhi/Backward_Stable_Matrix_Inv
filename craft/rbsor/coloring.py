"""Greedy BFS 2-coloring of the resistor graph.

Used by the RB-SOR solver to split nodes into two sweeps.

Convention: node 0 (GND) is always red. For disconnected components,
the smallest unvisited node is colored red and a fresh BFS is started.

For non-bipartite graphs (odd cycles, e.g. Wheatstone bridges with triangles),
some edges remain monochromatic; we return them as `conflict_edges` so the
caller can route those neighbors to the previous-iteration v-emission.
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

from collections import deque

from parse import ParsedCircuit


def two_color(pc: ParsedCircuit) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Greedy BFS 2-coloring.

    Returns:
        red_order:      list of node ids colored red, sorted ascending
        black_order:    list of node ids colored black, sorted ascending
        conflict_edges: deduplicated list of (min, max) edges where both
                        endpoints share a color
    """
    n = pc.num_nodes

    adj: list[set[int]] = [set() for _ in range(n)]
    for src in range(n):
        for dst, _w in pc.out_edges_norm[src]:
            adj[src].add(dst)
            adj[dst].add(src)

    color: list[int] = [-1] * n  # 0 = red, 1 = black, -1 = unvisited

    for seed in range(n):
        if color[seed] != -1:
            continue
        color[seed] = 0
        q: deque[int] = deque([seed])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    q.append(v)

    conflict_set: set[tuple[int, int]] = set()
    for src in range(n):
        for dst in adj[src]:
            if color[src] == color[dst] and src < dst:
                conflict_set.add((src, dst))

    red_order = sorted(i for i in range(n) if color[i] == 0)
    black_order = sorted(i for i in range(n) if color[i] == 1)
    return red_order, black_order, sorted(conflict_set)


def color_idx_map(red_order: list[int], black_order: list[int]) -> list[int]:
    """Return list `cidx` such that cidx[node] = position in red_order ++ black_order.

    Used by the tokenizer and DSL builder to compute neighbor offsets.
    """
    n = len(red_order) + len(black_order)
    cidx = [-1] * n
    for i, node in enumerate(red_order):
        cidx[node] = i
    base = len(red_order)
    for j, node in enumerate(black_order):
        cidx[node] = base + j
    return cidx


if __name__ == "__main__":
    import json
    import os
    import sys

    from parse import parse_netlist

    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    )
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            red, black, conflicts = two_color(pc)
            print(f"{c['ID']:<10} N={pc.num_nodes:<3} "
                  f"|red|={len(red)} |black|={len(black)} conflicts={len(conflicts)}")
            if conflicts and len(conflicts) <= 8:
                print(f"           conflict edges: {conflicts}")
