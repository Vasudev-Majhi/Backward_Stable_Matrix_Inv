"""Dump non-zero entries of selected token embeddings for the n=5 trace model."""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa
from transformer_vm.model.weights import load_weights

bin_files = sorted(glob.glob("/tmp/trace/model_*_lu.bin"))
if not bin_files:
    print("no model in /tmp/trace"); sys.exit(1)
mp = bin_files[-1]
sc = json.load(open(mp + ".slots.json"))
named = sc["slot_of_named"]

m, toks, t2i = load_weights(mp)

def show(name):
    emb = m.tok.weight[t2i[name]]
    print(f"== {name} ==")
    rows = []
    for nm, s in named.items():
        v = float(emb[s].item())
        if abs(v) > 1e-9:
            rows.append((s, nm, v))
    rows.sort()
    for s, nm, v in rows:
        print(f"  slot {s:>2} ({nm:<14}) = {v:+.6f}")

for name in ("init_2", "fwd_0", "fwd_2", "bck_3", "bck_0"):
    show(name)
    print()
