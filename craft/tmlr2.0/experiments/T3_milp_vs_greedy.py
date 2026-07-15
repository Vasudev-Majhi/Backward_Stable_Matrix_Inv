"""T1.3 — MILP scheduler ablation against ASAP-greedy on the full program graph.

For each of the 154 circuits we already have a MILP-compiled model. This
script reconstructs the program graph the MILP saw and computes:

  (a) MILP-actual n_layers and d_model       (from the existing per-circuit CSV)
  (b) ASAP-greedy lower bound on n_layers    (via _min_layers from milp.py;
      this is exactly the longest-dependency-chain length with phase parity)
  (c) ASAP-greedy d_model under first-fit interval coloring on the SAME
      lifetime graph the MILP used (so the comparison is apples-to-apples)

The MILP is only "necessary" if it improves (a) or (c) over greedy ASAP.

Output: results/T3/milp_vs_greedy.csv with columns
    circuit, n_free, milp_layers, milp_d_model, asap_layers,
    asap_d_model_firstfit, asap_d_model_peak
"""
from __future__ import annotations
import os, sys, csv, json, time, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
# inversion2 first so its _bootstrap is picked.
sys.path.insert(0, os.path.join(HOME, "craft_release", "craft"))
sys.path.insert(0, os.path.join(HOME, "craft_release", "matrix_inversion", "inversion2"))

os.environ.setdefault("MINV_USE_STANDARD_CACHE", "1")
import numpy as np
import _bootstrap  # noqa: F401

from transformer_vm.scheduler.milp import _build_graph, _min_layers
from transformer_vm.graph.core import LookUp, ReGLUDimension, PersistDimension

# Parser for circuit netlists.
from parse import parse_netlist  # type: ignore

DATASET = os.path.join(HOME, "craft_release", "dataset", "circuit_dataset_rv.jsonl")


def load_circuits():
    circs = []
    with open(DATASET) as f:
        for line in f:
            c = json.loads(line)
            pc = parse_netlist(c["Netlist"])
            N = pc.num_nodes
            A = np.zeros((N, N), dtype=np.float64)
            for (a, b, ohms) in pc.resistors:
                g = 1.0 / ohms
                A[a, a] += g; A[b, b] += g
                A[a, b] -= g; A[b, a] -= g
            free = [i for i in range(N) if not pc.is_fixed[i]]
            if not free:
                continue
            A_FF = A[np.ix_(free, free)]
            circs.append({"cid": c["ID"], "complexity": c.get("Complexity", "?"),
                          "N": N, "n_free": len(free), "A_FF": A_FF})
    return circs


def asap_layers_for_graph(graph):
    """ASAP layer count = minimum possible (longest dep chain / 4 phases)."""
    return _min_layers(graph["ops"], graph["op_deps"])


def asap_d_model_firstfit(graph):
    """ASAP layer assignment + first-fit interval coloring over op-produced dims.

    Returns (n_layers, d_model_firstfit, peak_alive)
    """
    ops = graph["ops"]
    op_deps = graph["op_deps"]
    produced = graph["produced"]
    consumers = graph["consumers"]

    # ASAP phase assignment (mirrors _min_layers but stores per-op phase).
    phase = {}
    remaining = set(ops)
    while remaining:
        progress = False
        for op in list(remaining):
            if not all(p in phase for p in op_deps[op]):
                continue
            lo = max((phase[p] for p in op_deps[op]), default=-1) + 1
            if isinstance(op, LookUp):
                lo += (-lo) % 4
            elif isinstance(op, ReGLUDimension):
                lo += (2 - lo % 4 + 4) % 4
            else:
                lo += 0 if lo % 2 == 1 else 1
            phase[op] = lo
            remaining.discard(op)
            progress = True
        assert progress, "cycle"
    n_layers = max(phase.values()) // 4 + 1

    # Lifetime of each produced dim: birth = phase[producer], death = max
    # consumer phase (or end-of-last-layer if no consumer).
    end_phase = 4 * n_layers - 1
    dim_birth, dim_death = {}, {}
    all_dims = set()
    for op in ops:
        for d in produced.get(op, []):
            all_dims.add(d)
            dim_birth[d] = phase[op]
            cs = [c for c in consumers.get(d, []) if c in phase]
            dim_death[d] = max((phase[c] for c in cs), default=end_phase)

    # Peak alive count
    boundaries = sorted({phase[op] for op in ops} | {0, end_phase})
    peak = 0
    for b in boundaries:
        alive = sum(1 for d in all_dims if dim_birth[d] <= b <= dim_death[d])
        peak = max(peak, alive)

    # First-fit coloring
    events = []
    for d in all_dims:
        events.append((dim_birth[d], 0, d))   # birth
        events.append((dim_death[d] + 1, 1, d))  # release
    events.sort(key=lambda e: (e[0], e[1]))
    slot_free_at = {}
    slot_of = {}
    next_slot = 0
    for ev in events:
        t, kind, d = ev
        if kind == 0:
            chosen = None
            for s in range(next_slot):
                if slot_free_at.get(s, 0) <= t:
                    chosen = s; break
            if chosen is None:
                chosen = next_slot; next_slot += 1
            slot_of[d] = chosen
            slot_free_at[chosen] = dim_death[d] + 1
    d_model_firstfit = next_slot
    return n_layers, d_model_firstfit, peak


def build_lu_program_graph_for_matrix(A):
    """Materialize the program graph that the MILP would see, without
    invoking the MILP solve."""
    from minv_interpreter_lu import MinvMachineLU
    machine = MinvMachineLU(A)
    pg, _meta = machine.build()
    return pg


def main():
    out_dir = os.path.join(HERE, "..", "results", "T3")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "milp_vs_greedy.csv")

    circuits = load_circuits()

    # MILP-actual layer/d_model: call the live MILP via build_for_matrix(plan_only=True).
    from build_lu import build_for_matrix

    fields = ["circuit", "complexity", "n_free", "milp_layers", "milp_d_model",
              "asap_layers", "asap_d_model_firstfit", "asap_peak_alive",
              "layers_eq", "d_model_savings_pct"]
    rows = []
    t0 = time.time()
    for i, c in enumerate(circuits):
        cid = c["cid"]
        try:
            # Live MILP solve (this also reuses cached results internally).
            sched = build_for_matrix(c["A_FF"], plan_only=True)
            milp_L = int(sched["num_layers"])
            milp_D = int(sched["width"])
            pg = build_lu_program_graph_for_matrix(c["A_FF"])
            graph = _build_graph(pg.all_dims, pg.all_lookups, pg.inv_log_pos)
            asap_L = asap_layers_for_graph(graph)
            asap_L2, asap_D, peak = asap_d_model_firstfit(graph)
            assert asap_L == asap_L2, f"{asap_L} vs {asap_L2}"
            rows.append({
                "circuit": cid, "complexity": c["complexity"], "n_free": c["n_free"],
                "milp_layers": milp_L, "milp_d_model": milp_D,
                "asap_layers": asap_L, "asap_d_model_firstfit": asap_D,
                "asap_peak_alive": peak,
                "layers_eq": int(milp_L == asap_L),
                "d_model_savings_pct": 100.0 * (milp_D - asap_D) / max(1, milp_D),
            })
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(circuits)}] cid={cid} milp=({milp_L},{milp_D}) asap=({asap_L},{asap_D})  elapsed={time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  {cid}: ERROR {e!r}", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {out_csv}")

    # Aggregate stats
    if rows:
        n = len(rows)
        eq = sum(r["layers_eq"] for r in rows)
        avg_save = sum(r["d_model_savings_pct"] for r in rows) / n
        print(f"\nAggregate over {n} circuits:")
        print(f"  layers match (MILP == ASAP): {eq}/{n}  ({100*eq/n:.1f}%)")
        print(f"  d_model FirstFit savings vs MILP: avg {avg_save:.1f}%")


if __name__ == "__main__":
    main()
