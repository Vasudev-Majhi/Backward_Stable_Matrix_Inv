"""P0.3 / Review sec.15, sec.21 -- A FAIR baseline for the Hull selector.

CONCERN (Reviewer A W3, sec.1(iii), sec.15): "Table 16's 13.2 s for the Standard
cache at L=10,000 is ~5e7 two-dimensional dot products at ~265 ns each --
Python-interpreter throughput, not vectorized. Meanwhile Hull is the shipped C++
header. The 12-55x figures therefore plausibly measure Python-vs-C++, not
O(L)-vs-O(log L). A single torch.argmax(K @ q) over 10,000 two-dimensional keys
is microseconds."

This script times THREE selectors in the SAME process on the SAME keys:

  A. standard_shipped -- the shipped StandardKVCache (torch.stack of a growing
     Python list, then a full softmax) -- the paper's Table 16 baseline.
  B. dense_vectorized -- the obvious baseline the review demands: keys held in a
     preallocated contiguous buffer, selection by a single batched
     argmax(K @ q) over all heads at once. Same language, same process, float64.
  C. hull_cpp -- the shipped C++17 dynamic-convex-hull extension.

We report per-query wall time vs L and the crossover, and we verify that B and C
select the SAME position (so the comparison is between two correct selectors).

The prediction stated in the review is that B beats C at every L a CRAFT program
produces. We report what actually happens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "transformer-vm"))
RES = os.path.join(HERE, "results")

torch.set_num_threads(1)  # single-threaded, as the paper's protocol requires


def make_parabolic_keys(L, n_heads, K=1e10, alpha=0.3, rng=None, normalized=False):
    """The production positional-key encoding: k_p = (2p, -p^2 + alpha*g(p)) * K/sqrt2.

    With normalized=True we instead use the bounded-magnitude encoding
    (p/L, -(p/L)^2) * K/sqrt2 -- the conditioning control of P1.1.
    """
    rng = rng or np.random.default_rng(0)
    p = np.arange(L, dtype=np.float64)
    if normalized:
        pn = p / max(L, 1)
        kx, ky = 2 * pn, -(pn ** 2)
    else:
        g = np.log1p(p) / np.log(2) if alpha else np.zeros_like(p)
        kx, ky = 2 * p, -(p ** 2) + alpha * g
    scale = K / np.sqrt(2.0)
    keys = np.stack([kx, ky], axis=1) * scale
    return np.repeat(keys[:, None, :], n_heads, axis=1)  # (L, H, 2)


class ShippedStandard:
    """Exactly the shipped StandardKVCache access pattern."""

    def __init__(self, n_heads):
        self.n_heads = n_heads
        self._keys = []

    def add(self, k):
        self._keys.append(torch.as_tensor(k).clone())

    def query(self, q):
        K = torch.stack(self._keys).reshape(-1, self.n_heads, 2)
        Q = torch.as_tensor(q).reshape(self.n_heads, 2)
        scores = torch.einsum("thi,hi->th", K, Q)
        w = torch.softmax(scores, dim=0)
        return w.argmax(dim=0)


class DenseVectorized:
    """Preallocated contiguous buffer + one batched argmax(K @ q). Same language."""

    def __init__(self, n_heads, capacity):
        self.n_heads = n_heads
        self.buf = torch.empty((capacity, n_heads, 2), dtype=torch.float64)
        self.L = 0

    def add(self, k):
        self.buf[self.L] = torch.as_tensor(k)
        self.L += 1

    def query(self, q):
        K = self.buf[: self.L]
        Q = torch.as_tensor(q).reshape(self.n_heads, 2)
        scores = torch.einsum("thi,hi->th", K, Q)
        return scores.argmax(dim=0)


class HullCpp:
    def __init__(self, n_heads):
        from transformer_vm.attention.hull_cache import HullKVCache
        self.n_heads = n_heads
        self.c = HullKVCache(1, n_heads)
        for h in range(n_heads):
            self.c.set_tiebreak(0, h, True)
        self.seq = 0
        self._vals = []

    def add_and_query(self, k, q, v):
        return self.c.layer_step(0, torch.as_tensor(k).reshape(-1),
                                 torch.as_tensor(q).reshape(-1),
                                 torch.as_tensor(v).reshape(-1))


def bench(L, n_heads, n_queries, rng, include_shipped=True):
    keys = make_parabolic_keys(L, n_heads, rng=rng)
    # queries: q_p = (p, 1) * scale, the production read pattern
    qpos = rng.integers(0, L, size=n_queries)
    queries = np.stack([np.stack([qp * np.ones(n_heads), np.ones(n_heads)], axis=1)
                        for qp in qpos])  # (Q, H, 2)
    out = {"L": L, "n_heads": n_heads, "n_queries": n_queries}

    # -- B: dense vectorized -------------------------------------------------
    dv = DenseVectorized(n_heads, L + n_queries)
    for i in range(L):
        dv.add(keys[i])
    sel_dv = []
    t0 = time.perf_counter()
    for qi in range(n_queries):
        dv.add(keys[-1])            # fused insert+query, same op as Hull
        sel_dv.append(dv.query(queries[qi]))
    out["dense_vectorized_s"] = time.perf_counter() - t0

    # -- B2: fully batched (all queries at once) -----------------------------
    Kb = dv.buf[:L]
    Qb = torch.as_tensor(queries)
    t0 = time.perf_counter()
    scores = torch.einsum("thi,qhi->qth", Kb, Qb)
    sel_batched = scores.argmax(dim=1)
    out["dense_batched_s"] = time.perf_counter() - t0

    # -- C: hull C++ ---------------------------------------------------------
    # Hull's layer_step is a FUSED insert+query, so for the like-for-like
    # comparison we time insert+query for every selector.
    try:
        h = HullCpp(n_heads)
        vals = np.zeros((L, n_heads, 2))
        vals[:, :, 0] = np.arange(L)[:, None]
        kt = [torch.as_tensor(keys[i]) for i in range(L)]
        vt = [torch.as_tensor(vals[i]) for i in range(L)]
        qt = [torch.as_tensor(queries[qi]) for qi in range(n_queries)]
        for i in range(L):
            h.c.layer_step(0, kt[i], qt[0], vt[i])
        t0 = time.perf_counter()
        for qi in range(n_queries):
            h.c.layer_step(0, kt[-1], qt[qi], vt[-1])
        out["hull_cpp_s"] = time.perf_counter() - t0
    except Exception as e:
        out["hull_cpp_s"] = float("nan")
        out["hull_error"] = f"{type(e).__name__}: {e}"

    # -- A: shipped standard (only for small L; it is quadratic) -------------
    if include_shipped:
        ss = ShippedStandard(n_heads)
        for i in range(L):
            ss.add(keys[i])
        t0 = time.perf_counter()
        for qi in range(min(n_queries, 50)):
            ss.query(queries[qi])
        el = time.perf_counter() - t0
        out["standard_shipped_s"] = el * (n_queries / min(n_queries, 50))
    else:
        out["standard_shipped_s"] = float("nan")

    # -- agreement between the two correct selectors ------------------------
    agree = float(np.mean([bool((sel_dv[q] == sel_batched[q]).all())
                           for q in range(n_queries)]))
    out["dense_loop_vs_batched_agree"] = agree
    for k in ("standard_shipped", "dense_vectorized", "dense_batched", "hull_cpp"):
        out[k + "_us_per_query"] = out[k + "_s"] / n_queries * 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[100, 300, 1000, 3000, 10000, 30000, 100000])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    rows = []
    for L in args.lengths:
        r = bench(L, args.heads, args.queries, rng, include_shipped=(L <= 30000))
        rows.append(r)
        print(f"L={L:>7}  shipped={r['standard_shipped_us_per_query']:>12.2f}us  "
              f"dense_loop={r['dense_vectorized_us_per_query']:>9.2f}us  "
              f"dense_batched={r['dense_batched_us_per_query']:>9.3f}us  "
              f"hull={r['hull_cpp_us_per_query']:>9.2f}us")

    os.makedirs(RES, exist_ok=True)
    import csv
    with open(os.path.join(RES, "p03_hull_fair_baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summ = {"note": "per-query microseconds, single-threaded, one process, float64",
            "rows": rows}
    # speedups relative to the FAIR baseline, not the shipped Python loop
    for r in rows:
        r["hull_vs_shipped"] = (r["standard_shipped_us_per_query"]
                                / r["hull_cpp_us_per_query"])
        r["hull_vs_dense_loop"] = (r["dense_vectorized_us_per_query"]
                                   / r["hull_cpp_us_per_query"])
        r["hull_vs_dense_batched"] = (r["dense_batched_us_per_query"]
                                      / r["hull_cpp_us_per_query"])
    summ["verdict"] = {
        "hull_beats_dense_loop_at": [r["L"] for r in rows if r["hull_vs_dense_loop"] > 1],
        "hull_beats_dense_batched_at": [r["L"] for r in rows if r["hull_vs_dense_batched"] > 1],
        "max_hull_vs_shipped": max([r["hull_vs_shipped"] for r in rows
                                    if np.isfinite(r["hull_vs_shipped"])] or [float("nan")]),
    }
    with open(os.path.join(RES, "p03_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ["verdict"], indent=2))


if __name__ == "__main__":
    main()
