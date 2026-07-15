"""Tiny summarizer for N5_v2 results."""
import json, os, sys

base = sys.argv[1] if len(sys.argv) > 1 else "results/N5_v2"
for cid in ("CKT_0001", "CKT_0015", "CKT_0067", "CKT_0068"):
    p = os.path.join(base, f"N5_acdc_{cid}.json")
    if not os.path.exists(p):
        print(f"{cid}: (missing)")
        continue
    d = json.load(open(p))
    per_layer = {}
    for e in d["edges"]:
        per_layer.setdefault(e["layer"], 0)
        if not e["kept"]:
            per_layer[e["layer"]] += 1
    total_crit = sum(per_layer.values())
    total = d["n_layers"] * d["n_heads"]
    nf = d["n_free"]
    print(f"{cid}  n_free={nf}  layers={d['n_layers']}  heads_per_layer={d['n_heads']}  "
          f"total_heads={total}  critical={total_crit}  per_layer={per_layer}")
