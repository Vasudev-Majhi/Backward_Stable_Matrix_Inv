"""
Utility to load benchmark graph datasets as Laplacian systems (Ax = b).

Each graph is modeled as a resistor network:
  - Edge (i, j) with weight w  <-->  conductance G_ij = w  (resistance R = 1/w)
  - Laplacian L:  L[i,i] = sum of conductances at node i
                  L[i,j] = -G_ij  for i != j
  - Inject 1 A at source node, extract 1 A at sink node -> solve L_red * v = b_red

Usage:
    from load_benchmark import load_graph, build_laplacian, build_rhs

    G = load_graph("karate")          # dict with 'edges', 'n_nodes', 'name', etc.
    L, b, mask = build_laplacian(G, source=0, sink=33)
    v = np.linalg.solve(L[mask][:, mask], b[mask])
"""
import os
import numpy as np

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "karate": {
        "file": "karate.edges",
        "n_nodes": 34,
        "n_edges": 78,
        "domain": "Social Network",
        "description": "Zachary's Karate Club — friendships in a US university karate club (1977)",
        "citation": "W.W. Zachary, J. Anthropol. Res. 33, 452-473 (1977)",
        "default_source": 0,
        "default_sink": 33,
        "weighted": False,
    },
    "dolphins": {
        "file": "dolphins.edges",
        "n_nodes": 62,
        "n_edges": 159,
        "domain": "Biological / Animal Social Network",
        "description": "Bottlenose dolphin associations off Doubtful Sound, New Zealand",
        "citation": "Lusseau et al., Behav. Ecol. Sociobiol. 54, 396-405 (2003)",
        "default_source": 0,
        "default_sink": 33,
        "weighted": False,
    },
    "lesmis": {
        "file": "lesmis.edges",
        "n_nodes": 77,
        "n_edges": 254,
        "domain": "Literary / Humanities Network",
        "description": "Character co-appearances in Les Miserables (weighted by count)",
        "citation": "D.E. Knuth, The Stanford GraphBase (1993); JSON via vega/vega-datasets",
        "default_source": 11,   # Valjean (most central)
        "default_sink": 48,     # Gavroche
        "weighted": True,
    },
    "polbooks": {
        "file": "polbooks.edges",
        "n_nodes": 105,
        "n_edges": 441,
        "domain": "Recommendation / Political Science Network",
        "description": "US political books co-purchased on Amazon.com around 2004 election",
        "citation": "V. Krebs (2004), unpublished; available via SuiteSparse Newman/polbooks",
        "default_source": 0,
        "default_sink": 84,
        "weighted": False,
    },
}


def load_graph(name: str) -> dict:
    """Load a named benchmark graph. Returns dict with edges list and metadata."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    meta = DATASETS[name].copy()
    fpath = os.path.join(DATASET_DIR, meta["file"])
    edges = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            edges.append((u, v, w))
    meta["edges"] = edges
    meta["name"] = name
    return meta


def build_laplacian(graph: dict) -> np.ndarray:
    """Build the N x N weighted graph Laplacian matrix."""
    n = graph["n_nodes"]
    L = np.zeros((n, n), dtype=np.float64)
    for u, v, w in graph["edges"]:
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def build_rhs(n: int, source: int, sink: int, current: float = 1.0) -> np.ndarray:
    """Build RHS vector: inject +current at source, extract at sink."""
    b = np.zeros(n, dtype=np.float64)
    b[source] = +current
    b[sink] = -current
    return b


def solve_resistor_network(name: str, source: int = None, sink: int = None,
                            current: float = 1.0) -> dict:
    """
    Full pipeline: load graph, build Laplacian, ground sink node, solve for voltages.

    Returns dict with:
        'voltages': node voltages (sink grounded = 0)
        'effective_resistance': V(source) / current
        'graph': the loaded graph dict
        'source', 'sink': node indices used
    """
    graph = load_graph(name)
    if source is None:
        source = graph["default_source"]
    if sink is None:
        sink = graph["default_sink"]

    n = graph["n_nodes"]
    L = build_laplacian(graph)
    b = build_rhs(n, source, sink, current)

    # Ground the sink node: remove its row/col, solve reduced system
    keep = [i for i in range(n) if i != sink]
    L_red = L[np.ix_(keep, keep)]
    b_red = b[keep]

    v_red = np.linalg.solve(L_red, b_red)

    # Reconstruct full voltage vector (sink = 0)
    v = np.zeros(n)
    for ki, ni in enumerate(keep):
        v[ni] = v_red[ki]

    src_idx_in_keep = keep.index(source)
    R_eff = v_red[src_idx_in_keep] / current

    return {
        "voltages": v,
        "effective_resistance": R_eff,
        "graph": graph,
        "source": source,
        "sink": sink,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Benchmark Dataset Loader — Resistor Network Solver")
    print("=" * 60)
    for name, meta in DATASETS.items():
        result = solve_resistor_network(name)
        g = result["graph"]
        print(f"\n{name.upper()}")
        print(f"  Domain   : {g['domain']}")
        print(f"  Nodes    : {g['n_nodes']}  Edges: {g['n_edges']}")
        print(f"  Weighted : {g['weighted']}")
        print(f"  Source   : {result['source']}  Sink: {result['sink']}")
        print(f"  R_eff    : {result['effective_resistance']:.6f} Ohm")
        v = result["voltages"]
        print(f"  Voltages : min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
