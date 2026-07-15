"""Pure-Python Jacobi iteration for resistive networks.

Ground-truth reference. Matches the integer arithmetic the DSL will do
(voltages scaled by 10000, weights pre-normalized so Σ out = 10000 per node).
"""
from __future__ import annotations

from parse import ParsedCircuit, parse_netlist

SCALE = 10000


def jacobi_solve_parsed(pc: ParsedCircuit, target: int, T: int) -> float:
    """Run T Jacobi sweeps and return voltage at `target` in volts."""
    n = pc.num_nodes
    # Voltages in scaled integers.
    v = [int(round(pc.fixed_voltage[i] * SCALE)) if pc.is_fixed[i] else 0
         for i in range(n)]

    for _ in range(T):
        v_new = [0] * n
        for node in range(n):
            if pc.is_fixed[node]:
                v_new[node] = int(round(pc.fixed_voltage[node] * SCALE))
                continue
            # v_new = Σ (w_norm * v_neighbor) / 10000
            acc = 0
            for nb, w in pc.out_edges_norm[node]:
                acc += w * v[nb]
            v_new[node] = acc // SCALE
        v = v_new

    return v[target] / SCALE


def jacobi_solve(netlist: str, target: int, T: int) -> float:
    pc = parse_netlist(netlist)
    return jacobi_solve_parsed(pc, target, T)


def _auto_T(n_nodes: int) -> int:
    return max(1000, 50 * n_nodes)


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/circuit_dataset_rv.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    tol = 0.05

    passed = 0
    failed: list[tuple[str, float, float]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            T = _auto_T(pc.num_nodes)
            got = jacobi_solve_parsed(pc, int(c["Target_Node"]), T)
            truth = c["Ground_Truth_Vout"]
            ok = abs(got - truth) < tol
            status = "PASS" if ok else "FAIL"
            print(f"{status} {c['ID']:<10} T={T:<5} got={got:.4f} truth={truth:.4f} N={pc.num_nodes}")
            if ok:
                passed += 1
            else:
                failed.append((c["ID"], got, truth))

    print(f"\n{passed}/{min(limit, i + 1)} passed")
    if failed:
        for fid, g, t in failed:
            print(f"  FAIL {fid}: got {g:.4f} truth {t:.4f} (err {abs(g - t):.4f})")
