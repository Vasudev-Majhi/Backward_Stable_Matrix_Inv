"""SPICE netlist parser for resistor + voltage-source circuits.

Produces a ParsedCircuit with integer node IDs (0 = GND), fixed-node voltages,
and resistor edges. Shared by the Jacobi reference and the tokenizer so
both see identical structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_SI_SUFFIX = {
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}


def _parse_number(tok: str) -> float:
    """Parse a SPICE numeric with optional SI suffix. Case-insensitive."""
    t = tok.lower().strip()
    for suf in ("meg", "k", "m", "u", "n", "p"):
        if t.endswith(suf):
            return float(t[: -len(suf)]) * _SI_SUFFIX[suf]
    return float(t)


@dataclass
class ParsedCircuit:
    num_nodes: int
    is_fixed: list[bool]            # per node
    fixed_voltage: list[float]      # volts; 0.0 for free nodes
    resistors: list[tuple[int, int, float]]  # (n1, n2, ohms)

    # derived at parse time
    degree: list[int] = field(default_factory=list)
    # per-node normalized outgoing conductances, scaled so each sum = 10000
    # stored as list[list[(neighbor, w_norm_int)]]
    out_edges_norm: list[list[tuple[int, int]]] = field(default_factory=list)

    def edges_directed(self) -> list[tuple[int, int, int]]:
        """Yield (src, dst, w_norm_int) for every directed edge."""
        out = []
        for src, adj in enumerate(self.out_edges_norm):
            for dst, w in adj:
                out.append((src, dst, w))
        return out


def parse_netlist(netlist: str) -> ParsedCircuit:
    """Parse a SPICE-like netlist. GND is node 0. V-sources must connect to GND."""
    nodes: set[int] = {0}
    fixed: dict[int, float] = {}
    resistors: list[tuple[int, int, float]] = []

    for raw in netlist.splitlines():
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        head = parts[0]
        kind = head[0].upper()

        if kind == "V":
            # V<name> N+ N- <volts>
            if len(parts) < 4:
                raise ValueError(f"malformed V line: {raw!r}")
            n_plus, n_minus = int(parts[1]), int(parts[2])
            volts = _parse_number(parts[3])
            if n_minus != 0:
                raise ValueError(
                    f"V-source {head} does not connect to GND "
                    f"(N- = {n_minus}); only GND-referenced sources supported"
                )
            nodes.add(n_plus)
            # If the same node is fixed twice, last wins — warn by raising.
            if n_plus in fixed and fixed[n_plus] != volts:
                raise ValueError(f"node {n_plus} fixed to two voltages")
            fixed[n_plus] = volts

        elif kind == "R":
            # R<name> N1 N2 <ohms>
            if len(parts) < 4:
                raise ValueError(f"malformed R line: {raw!r}")
            n1, n2 = int(parts[1]), int(parts[2])
            ohms = _parse_number(parts[3])
            if ohms <= 0:
                raise ValueError(f"non-positive resistance in {raw!r}")
            nodes.add(n1)
            nodes.add(n2)
            resistors.append((n1, n2, ohms))

        else:
            # Ignore other element types (this dataset only has V + R).
            continue

    if not nodes or max(nodes) + 1 != len(nodes):
        # Nodes must be 0..N-1 contiguous. Re-map if not.
        sorted_ids = sorted(nodes)
        remap = {old: new for new, old in enumerate(sorted_ids)}
        if remap[0] != 0:
            raise ValueError("GND must be node 0 in the netlist")
        resistors = [(remap[a], remap[b], r) for (a, b, r) in resistors]
        fixed = {remap[n]: v for n, v in fixed.items()}
        n = len(sorted_ids)
    else:
        n = max(nodes) + 1

    is_fixed = [False] * n
    fixed_v = [0.0] * n
    is_fixed[0] = True  # GND
    fixed_v[0] = 0.0
    for node, v in fixed.items():
        is_fixed[node] = True
        fixed_v[node] = v

    # Degree and normalized outgoing conductances per node.
    deg = [0] * n
    sum_g = [0.0] * n
    adj_g: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, ohms in resistors:
        g = 1.0 / ohms
        adj_g[a].append((b, g))
        adj_g[b].append((a, g))
        sum_g[a] += g
        sum_g[b] += g
        deg[a] += 1
        deg[b] += 1

    out_norm: list[list[tuple[int, int]]] = []
    for node in range(n):
        if sum_g[node] == 0.0:
            out_norm.append([])
            continue
        scale = 10000.0 / sum_g[node]
        # Merge parallel edges (same neighbor) into one normalized weight.
        merged: dict[int, float] = {}
        for nb, g in adj_g[node]:
            merged[nb] = merged.get(nb, 0.0) + g * scale
        out_norm.append([(nb, int(round(w))) for nb, w in merged.items()])

    return ParsedCircuit(
        num_nodes=n,
        is_fixed=is_fixed,
        fixed_voltage=fixed_v,
        resistors=resistors,
        degree=deg,
        out_edges_norm=out_norm,
    )
