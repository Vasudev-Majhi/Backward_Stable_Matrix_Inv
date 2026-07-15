"""Analyze CADJ vs RB-SOR results and print head-to-head comparison."""
import csv
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CADJ_RESULTS = os.path.join(BASE, "cadj_results")
RBSOR_RESULTS = os.path.join(BASE, "rbsor_results")


def load_csv(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r.get("circuit_id", "").strip()
            if cid.startswith("CKT_"):
                rows[cid] = r
    return rows


def main():
    cadj_c1  = load_csv(os.path.join(CADJ_RESULTS,  "results_main_cadj.csv"))
    cadj_v01 = load_csv(os.path.join(CADJ_RESULTS,  "results_main_cadj_v01.csv"))
    rbsor_v0 = load_csv(os.path.join(RBSOR_RESULTS, "results_main_rbsor.csv"))
    rbsor_v01= load_csv(os.path.join(RBSOR_RESULTS, "results_main_rbsor_v01.csv"))

    print(f"Loaded: cadj_c1={len(cadj_c1)}, cadj_v01={len(cadj_v01)}, "
          f"rbsor_v0={len(rbsor_v0)}, rbsor_v01={len(rbsor_v01)}")

    # --- C1 vs v0 same-step comparison ---
    cadj_c1_pass  = sum(1 for r in cadj_c1.values()  if r.get("pass_fail") == "PASS")
    rbsor_v0_pass = sum(1 for r in rbsor_v0.values() if r.get("pass_fail") == "PASS")
    cadj_v01_pass = sum(1 for r in cadj_v01.values() if r.get("pass_fail") == "PASS")
    rbsor_v01_pass= sum(1 for r in rbsor_v01.values()if r.get("pass_fail") == "PASS")

    print()
    print("=== Same-step comparison ===")
    print(f"  V_STEP=0.05:  CADJ {cadj_c1_pass}/154  vs  RB-SOR {rbsor_v0_pass}/154  (+{cadj_c1_pass - rbsor_v0_pass})")
    print(f"  V_STEP=0.01:  CADJ {cadj_v01_pass}/154  vs  RB-SOR {rbsor_v01_pass}/154  (+{cadj_v01_pass - rbsor_v01_pass})")

    # --- Head-to-head v01 ---
    all_ids = sorted(set(cadj_v01.keys()) | set(rbsor_v01.keys()))
    rescue, regress, both_pass, both_fail = [], [], [], []
    for cid in all_ids:
        c = cadj_v01.get(cid, {})
        r = rbsor_v01.get(cid, {})
        cp = c.get("pass_fail") == "PASS"
        rp = r.get("pass_fail") == "PASS"
        if cp and rp:   both_pass.append(cid)
        elif cp:        rescue.append((cid, c, r))
        elif rp:        regress.append((cid, c, r))
        else:           both_fail.append((cid, c, r))

    print()
    print("=== CADJ v01 vs RB-SOR v01 (V_STEP=0.01) — bucket breakdown ===")
    print(f"  Both PASS:        {len(both_pass)}")
    print(f"  CADJ rescues:     {len(rescue)}  (RB-SOR fail -> CADJ pass)")
    print(f"  CADJ regressions: {len(regress)}  (RB-SOR pass -> CADJ fail)")
    print(f"  Both FAIL:        {len(both_fail)}")

    if rescue:
        print()
        print("--- CADJ RESCUES ---")
        for cid, c, r in sorted(rescue):
            print(f"  {cid:12} N={c.get('N','?'):3}  cadj_err={c.get('abs_error','?'):7}  "
                  f"rbsor_err={r.get('abs_error','?'):7}  {c.get('complexity','?')}")

    if regress:
        print()
        print("--- CADJ REGRESSIONS ---")
        for cid, c, r in sorted(regress):
            print(f"  {cid:12} N={c.get('N','?'):3}  cadj_err={c.get('abs_error','?'):7}  "
                  f"rbsor_err={r.get('abs_error','?'):7}  {c.get('complexity','?')}")

    if both_fail:
        print()
        print("--- BOTH FAIL ---")
        for cid, c, r in sorted(both_fail):
            ce = c.get('abs_error','?')
            re = r.get('abs_error','?')
            # Who is closer?
            try:
                delta = float(ce) - float(re)
                who = f"cadj_better={delta:.4f}" if delta < 0 else f"rbsor_better={-delta:.4f}"
            except Exception:
                who = ""
            print(f"  {cid:12} N={c.get('N','?'):3}  cadj_err={ce:7}  rbsor_err={re:7}  {who}  {c.get('complexity','?')}")

    # Write diff CSV
    diff_csv = os.path.join(CADJ_RESULTS, "cadj_vs_rbsor_diff.csv")
    fieldnames = ["circuit_id","complexity","N","bucket",
                  "cadj_err","rbsor_err","better_by",
                  "cadj_pred","rbsor_pred","truth_V"]
    with open(diff_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for cid in all_ids:
            c = cadj_v01.get(cid, {})
            r = rbsor_v01.get(cid, {})
            cp = c.get("pass_fail") == "PASS"
            rp = r.get("pass_fail") == "PASS"
            bucket = ("both_pass" if (cp and rp) else
                      "cadj_rescue" if cp else
                      "cadj_regress" if rp else "both_fail")
            try:
                better = float(r.get("abs_error","nan")) - float(c.get("abs_error","nan"))
            except Exception:
                better = ""
            w.writerow({
                "circuit_id": cid,
                "complexity": c.get("complexity", r.get("complexity","")),
                "N": c.get("N", r.get("N","")),
                "bucket": bucket,
                "cadj_err": c.get("abs_error",""),
                "rbsor_err": r.get("abs_error",""),
                "better_by": f"{better:.4f}" if isinstance(better, float) else better,
                "cadj_pred": c.get("pred_V_dsl",""),
                "rbsor_pred": r.get("pred_V_rbsor", r.get("pred_V_dsl","")),
                "truth_V": c.get("truth_V", r.get("truth_V","")),
            })
    print(f"\nDiff CSV written: {diff_csv}")


if __name__ == "__main__":
    main()
