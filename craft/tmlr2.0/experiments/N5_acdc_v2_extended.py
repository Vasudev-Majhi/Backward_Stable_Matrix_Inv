"""N5_v2 extension: add CKT_0084 (n_free=17) and CKT_0086 (n_free=19).

Reuses N5_acdc_v2.run_one_circuit verbatim — only changes the circuit list.
Doubles the L2=0 generalisation evidence by pushing to n_free=17 and 19
(2x and 2.4x larger than the existing CKT_0068 ceiling at n_free=8).

Output: results/N5_v2/N5_acdc_CKT_0084.json
        results/N5_v2/N5_acdc_CKT_0086.json
        results/N5_v2/N5_acdc_summary_extended.csv
"""
from __future__ import annotations
import os, sys, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from N5_acdc_v2 import run_one_circuit


def main():
    out_dir = os.path.join(HERE, "..", "results", "N5_v2")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("/tmp/N5_v2_models", exist_ok=True)

    # New circuits, sorted small-to-large so the easier one finishes first.
    circuits = ["CKT_0084", "CKT_0086"]
    summary_rows = []
    for cid in circuits:
        try:
            row = run_one_circuit(cid, out_dir)
            summary_rows.append(row)
        except Exception as e:
            print(f"  FAILED on {cid}: {e!r}", flush=True)
            import traceback; traceback.print_exc()

    out_csv = os.path.join(out_dir, "N5_acdc_summary_extended.csv")
    if summary_rows:
        with open(out_csv, "w", newline="") as f:
            keys = [k for k in summary_rows[0].keys() if k != "per_layer_critical"]
            keys += ["per_layer_critical"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in summary_rows:
                r2 = dict(r); r2["per_layer_critical"] = json.dumps(r["per_layer_critical"])
                w.writerow(r2)
        print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
