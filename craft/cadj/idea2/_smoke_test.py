import sys, os, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CADJ = os.path.dirname(_HERE)
_CF   = os.path.dirname(_CADJ)
for p in (_CF, _CADJ, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa

from build_idea2 import build_for_circuit
from run_idea2 import run_circuit_idea2

circuits = ["CKT_0001", "CKT_0034", "CKT_0098", "CKT_0143"]
results = []
for cid in circuits:
    t0 = time.time()
    build_for_circuit(cid)   # cached, instant
    status, pred_v, truth, _ = run_circuit_idea2(cid)
    err = abs(pred_v - truth)
    dt = time.time() - t0
    flag = "PASS" if status == "PASS" else "FAIL"
    print(f"{flag}  {cid}  pred={pred_v:.4f}  truth={truth:.4f}  err={err:.4f}  {dt:.1f}s", flush=True)
    results.append(flag)

passed = results.count("PASS")
print(f"\nSmoke test: {passed}/{len(results)} PASS")
