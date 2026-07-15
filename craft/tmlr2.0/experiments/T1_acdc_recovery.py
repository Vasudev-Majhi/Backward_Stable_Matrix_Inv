"""T1.1 — ACDC-style edge-recovery validation on CRAFT models.

Combines two existing analyses to produce the "ACDC recovers the known LU
dependency graph" demonstration the audit demanded:

  Step 1: head ablation (already done in N5_v2) identifies the set of
          critical heads via the same greedy keep/drop criterion ACDC uses.
  Step 2: attention argmax extraction (already done in interp_t1.py) gives
          the (query_tok, key_tok) edges each head implements.
  Step 3: this script joins the two and compares against the analytic LU
          dependency DAG.

LU dependency DAG (token-level):
  - fwd_i token's attention must read fwd_j tokens (j < i) to fetch y_j
  - bck_i token's attention must read bck_j tokens (j > i) to fetch x_j
  - All other (query, key) pairs are bookkeeping (over-provisioned by the
    MILP scheduler).

Recovery metrics per circuit:
  - critical_head_recall:  fraction of LU-edges hit by at least one
                            (critical_head, query_pos, argmax_key_pos)
  - critical_head_precision: fraction of (critical_head, query, argmax_key)
                              triples that match a real LU edge
  - non_critical_head_lu_hits: of the "silent" heads, how many still have
                                LU-aligned argmax (purely diagnostic)

Output: results/T1_acdc/T1_recovery_summary.csv
        results/T1_acdc/T1_recovery_<cid>.json

This is the canonical ACDC-recovers-LU demonstration the paper claims as MI
ground-truth value.
"""
from __future__ import annotations
import os, sys, csv, json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")

# Inputs
N5V2_DIR = os.path.join(HOME, "craft_release", "tmlr2_runs", "results", "N5_v2")
ATTN_CSV = os.path.join(HOME, "craft_release", "craft",
                        "paper_experiments", "results", "attention_patterns_T1.csv")
# Output
OUT_DIR = os.path.join(HOME, "craft_release", "tmlr2_runs", "results", "T1_acdc")


def lu_groundtruth_token_edges(n_free):
    """LU dependency DAG at the token-position level.

    Token sequence per column j (length 3n+2):
      pos 0       : start
      pos 1..n    : init_0 .. init_{n-1}
      pos n+1..2n : fwd_0 .. fwd_{n-1}
      pos 2n+1..3n: bck_{n-1} .. bck_0 (REVERSE order)
      pos 3n+1    : halt

    Required edges (query_pos, key_pos):
      fwd_i (pos = n+1+i) -> fwd_j (pos = n+1+j) for j < i
      bck_i (pos = 2n+1 + (n-1-i)) -> bck_j (pos = 2n+1 + (n-1-j)) for j > i
    """
    n = n_free
    edges = set()
    for i in range(n):
        for j in range(i):
            edges.add((n + 1 + i, n + 1 + j, "fwd_y"))
        for j in range(i + 1, n):
            q = 2 * n + 1 + (n - 1 - i)
            k = 2 * n + 1 + (n - 1 - j)
            edges.add((q, k, "bck_x"))
    return edges


def load_critical_heads(n5_path):
    d = json.load(open(n5_path))
    crit = set()
    for e in d["edges"]:
        if not e["kept"]:
            crit.add((e["layer"], e["head"]))
    return crit, d["n_free"], d["n_layers"], d["n_heads"]


def load_attention_for_circuit(cid):
    """Return list of dicts: layer, head, query_pos, argmax_key_pos, entropy."""
    rows = []
    if not os.path.exists(ATTN_CSV):
        return rows
    with open(ATTN_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("circuit") != cid or r.get("transformer") != "T1":
                continue
            rows.append({
                "layer": int(r["layer"]),
                "head": int(r["head"]),
                "query_pos": int(r["query_pos"]),
                "query_tok": r.get("query_tok", ""),
                "argmax_key_pos": int(r["argmax_key_pos"]),
                "max_score": float(r["max_score"]),
                "attn_entropy": float(r.get("attn_entropy", 0.0)),
            })
    return rows


def analyse_circuit(cid):
    n5_path = os.path.join(N5V2_DIR, f"N5_acdc_{cid}.json")
    if not os.path.exists(n5_path):
        return None
    crit, n_free, n_layers, n_heads_per = load_critical_heads(n5_path)
    attn_rows = load_attention_for_circuit(cid)
    if not attn_rows:
        return {"circuit": cid, "n_free": n_free,
                "n_layers": n_layers, "n_heads_per_layer": n_heads_per,
                "total_heads": n_layers * n_heads_per,
                "critical_count": len(crit),
                "lu_edges_total": len(lu_groundtruth_token_edges(n_free)),
                "critical_recall": -1.0,
                "critical_precision": -1.0,
                "critical_total_triples": 0,
                "critical_alignment": 0,
                "silent_lu_aligned": 0,
                "silent_total_triples": 0,
                "attn_available": False}

    lu_edges = lu_groundtruth_token_edges(n_free)
    # Strip the third element of edges (just (q, k))
    lu_edge_set = {(q, k) for (q, k, _) in lu_edges}

    # For each (layer, head, query_pos) collect the argmax key.
    critical_attn_pairs = set()   # (query_pos, argmax_key_pos) from critical heads
    silent_attn_pairs = set()
    crit_alignment = 0
    crit_total_triples = 0
    silent_alignment = 0
    silent_total_triples = 0
    for r in attn_rows:
        lh = (r["layer"], r["head"])
        qk = (r["query_pos"], r["argmax_key_pos"])
        is_crit = lh in crit
        if is_crit:
            critical_attn_pairs.add(qk)
            crit_total_triples += 1
            if qk in lu_edge_set:
                crit_alignment += 1
        else:
            silent_attn_pairs.add(qk)
            silent_total_triples += 1
            if qk in lu_edge_set:
                silent_alignment += 1

    # Recall: fraction of LU edges hit by at least one critical-head (q, k).
    recall = len(lu_edge_set & critical_attn_pairs) / max(1, len(lu_edge_set))
    crit_precision = crit_alignment / max(1, crit_total_triples)

    return {
        "circuit": cid, "n_free": n_free,
        "n_layers": n_layers, "n_heads_per_layer": n_heads_per,
        "total_heads": n_layers * n_heads_per,
        "critical_count": len(crit),
        "lu_edges_total": len(lu_edge_set),
        "critical_recall": recall,
        "critical_precision": crit_precision,
        "critical_total_triples": crit_total_triples,
        "critical_alignment": crit_alignment,
        "silent_lu_aligned": silent_alignment,
        "silent_total_triples": silent_total_triples,
        "attn_available": True,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "T1_recovery_summary.csv")
    circuits = ["CKT_0001", "CKT_0015", "CKT_0067", "CKT_0068"]
    rows = []
    for cid in circuits:
        r = analyse_circuit(cid)
        if r is None:
            print(f"  {cid}: no N5_v2 result yet"); continue
        rows.append(r)
        with open(os.path.join(OUT_DIR, f"T1_recovery_{cid}.json"), "w") as f:
            json.dump(r, f, indent=2)
        if r["attn_available"]:
            print(f"  {cid}: critical={r['critical_count']}/{r['total_heads']}  "
                  f"LU_edges={r['lu_edges_total']}  recall={r['critical_recall']:.2%}  "
                  f"crit_precision={r['critical_precision']:.2%}")
        else:
            print(f"  {cid}: critical={r['critical_count']}/{r['total_heads']}  "
                  f"LU_edges={r['lu_edges_total']}  (no attention CSV for this cid)")
    if rows:
        fields = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
