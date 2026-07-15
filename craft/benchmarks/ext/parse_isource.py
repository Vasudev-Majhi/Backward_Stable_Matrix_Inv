"""SPICE netlist parser extended with current-source (I) elements.

This wraps `parse.parse_netlist` (existing, untouched) without modifying it.
It first feeds the netlist through the existing parser to obtain the V+R
structure, then runs a second pass to extract I-source lines, and produces a
`ParsedCircuitWithI` dataclass that adds a per-node `i_inj_norm` field.

I-source convention:
    I<name> N+ N- <amperes>
    - N- must be 0 (GND).
    - Positive value = current INTO node N+ from the external source
      (i.e., current injection into N+).

In the Jacobi nodal update with conductances pre-normalized to Σ w = SCALE:
    V_new[n] = (Σ w_k · V_nbr[k]) / SCALE        (existing V+R-only)
    V_new[n] = (Σ w_k · V_nbr[k]) / SCALE + I_inj_norm[n]   (with I-sources)

where I_inj_norm[n] = I_inj_raw[n] / Σ_g_raw[n]  is the equivalent voltage
contribution from external current at node n. We pre-compute this as scaled
integer volts (I_inj_norm_scaled = round(I_inj_norm * SCALE)) at parse time
so the interpreter can bake it into the update-token embedding.
"""
from __future__ import annotations

import benchmarks._path  # noqa: F401

from dataclasses import dataclass, field

from parse import ParsedCircuit, _parse_number, parse_netlist

SCALE = 10000  # mirrors parse.py's normalization constant


@dataclass
class ParsedCircuitWithI(ParsedCircuit):
    """ParsedCircuit + per-node current injection (volts after normalization).

    i_inj_norm[n] is the equivalent voltage contribution from external current
    sources at node n (already divided by Σ raw conductance at that node).
    Zero for nodes with no I-sources or for fixed nodes (V-source nodes).
    """
    i_inj_norm: list[float] = field(default_factory=list)
    # Raw injected current per node, in amperes (for diagnostics).
    i_inj_raw: list[float] = field(default_factory=list)


def parse_netlist_isource(netlist: str) -> ParsedCircuitWithI:
    """Parse a netlist that may contain `I` lines.

    Returns a ParsedCircuitWithI even if no I-sources are present (in which
    case i_inj_norm is all zeros — semantics identical to the standard parser).
    """
    pc = parse_netlist(netlist)

    # Compute raw conductance sums per node (mirrors parse.py's internal
    # state — we reproduce it here since parse.py doesn't expose it).
    sum_g_raw: list[float] = [0.0] * pc.num_nodes
    for a, b, ohms in pc.resistors:
        g = 1.0 / ohms
        sum_g_raw[a] += g
        sum_g_raw[b] += g

    # Second pass over the netlist: scan I-lines.
    # Same node-remapping caveat: parse_netlist re-maps nodes if they aren't
    # contiguous 0..N-1. For our generators (heat/Poisson, IEEE) the nodes
    # are always already contiguous, so remap is identity. To stay safe, we
    # rebuild the sorted-id remap here exactly as parse.py does.
    nodes_in_text: set[int] = {0}
    raw_lines: list[tuple[int, int, float]] = []  # (n_plus, n_minus, amperes)

    for raw in netlist.splitlines():
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        head = parts[0]
        kind = head[0].upper()
        if kind == "I":
            if len(parts) < 4:
                raise ValueError(f"malformed I line: {raw!r}")
            n_plus, n_minus = int(parts[1]), int(parts[2])
            amps = _parse_number(parts[3])
            if n_minus != 0:
                raise ValueError(
                    f"I-source {head} does not connect to GND "
                    f"(N- = {n_minus}); only GND-referenced injections supported"
                )
            nodes_in_text.add(n_plus)
            raw_lines.append((n_plus, n_minus, amps))
        elif kind == "V":
            if len(parts) >= 4:
                nodes_in_text.add(int(parts[1]))
                nodes_in_text.add(int(parts[2]))
        elif kind == "R":
            if len(parts) >= 4:
                nodes_in_text.add(int(parts[1]))
                nodes_in_text.add(int(parts[2]))

    # Verify same remap as parse.py.
    if not nodes_in_text or max(nodes_in_text) + 1 != len(nodes_in_text):
        sorted_ids = sorted(nodes_in_text)
        remap = {old: new for new, old in enumerate(sorted_ids)}
    else:
        remap = {n: n for n in nodes_in_text}

    i_inj_raw = [0.0] * pc.num_nodes
    for (n_plus, _n_minus, amps) in raw_lines:
        if n_plus in remap:
            n = remap[n_plus]
            if 0 <= n < pc.num_nodes:
                i_inj_raw[n] += amps

    # Normalize per node: I_inj_norm[n] = I_inj_raw[n] / sum_g_raw[n]
    # (in volts). Skip fixed nodes — their voltage is forced by V-source so any
    # injection there is irrelevant to the iterative update.
    i_inj_norm = [0.0] * pc.num_nodes
    for n in range(pc.num_nodes):
        if pc.is_fixed[n]:
            continue
        if sum_g_raw[n] == 0.0:
            if i_inj_raw[n] != 0.0:
                raise ValueError(
                    f"node {n} has I-source injection but no resistive path "
                    f"(disconnected free node)"
                )
            continue
        i_inj_norm[n] = i_inj_raw[n] / sum_g_raw[n]

    return ParsedCircuitWithI(
        num_nodes=pc.num_nodes,
        is_fixed=pc.is_fixed,
        fixed_voltage=pc.fixed_voltage,
        resistors=pc.resistors,
        degree=pc.degree,
        out_edges_norm=pc.out_edges_norm,
        i_inj_norm=i_inj_norm,
        i_inj_raw=i_inj_raw,
    )


if __name__ == "__main__":
    test_netlist = """
* 4-node test: I-source + R + V
V1 1 0 10.0
R1 1 2 1k
R2 2 3 1k
R3 3 0 1k
I1 2 0 0.005
.op
.end
""".strip()
    pc = parse_netlist_isource(test_netlist)
    print(f"N={pc.num_nodes}")
    print(f"is_fixed = {pc.is_fixed}")
    print(f"fixed_voltage = {pc.fixed_voltage}")
    print(f"resistors = {pc.resistors}")
    print(f"i_inj_raw = {pc.i_inj_raw}")
    print(f"i_inj_norm = {pc.i_inj_norm}")
