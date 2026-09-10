# `revision/` — experiments answering the adversarial review

Everything here runs on **CPU in float64**. No GPU is used or needed: float64 is
mandatory for this pipeline (the attention scale overflows in float32), and the
workload is interpreter-dispatch-bound rather than FLOP-bound.

Start with **[`RESPONSE_TO_REVIEWERS.md`](RESPONSE_TO_REVIEWERS.md)** — the
point-by-point response with every number in context.

## Environment

Built and run on macOS (Apple M5 Pro, 15 cores), Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install mpmath
export CRAFT_DATASET="$PWD/dataset/circuit_dataset_rv.jsonl"
```

Resolved versions: torch 2.14.0, numpy 2.4.6, scipy 1.17.1, PuLP with HiGHS
1.15.1 (the current PuLP default; CBC is also exercised in `p07`).

Note this differs from the paper's pinned environment (Python 3.12.3, torch
2.6.0+cu124, numpy 2.4.4, PuLP 3.3.0/CBC). Results here are internally pinned and
self-consistent; cross-environment byte identity is precisely what `p07`
investigates, and it does **not** hold unconditionally.

## Scripts

| Script | Review item | Output |
|---|---|---|
| `craft_harness.py` | — | Shared harness: run-time-RHS driver + instrumented attention cache |
| `p01_runtime_rhs.py` | **P0.1** | `results/p01_summary.json`, `p01_runtime_rhs.csv` |
| `p02_margin_bound.py` | **P0.2** | `results/p02_margin_bound.json` |
| `p03_hull_fair_baseline.py` | **P0.3** | `results/p03_hull_fair_baseline.csv`, `p03_summary.json` |
| `p05_voltage_decomposition.py` | **P0.5** | `results/p05_summary.json`, `p05_voltage_decomposition.csv` |
| `p07_schedule_stability.py` | **P0.7** | `results/p07_summary.json`, `p07_schedule_stability.csv` |
| `p11_proxy_diagnosticity.py` | **P1.1–P1.4** | `results/p11_proxy_grid.csv` |
| `p11b_analyze_and_figure.py` | **P1.1**, §24 | `results/p11_diagnosticity.json`, `figures/killer_figure.png` |

## Reproduce

```bash
.venv/bin/python revision/p02_margin_bound.py            # seconds
.venv/bin/python revision/p05_voltage_decomposition.py   # ~1 min
.venv/bin/python revision/p03_hull_fair_baseline.py      # ~3 min
.venv/bin/python revision/p07_schedule_stability.py      # ~10 min
MILP_TIME_LIMIT=45 .venv/bin/python revision/p01_runtime_rhs.py   # ~40 min
MILP_TIME_LIMIT=30 .venv/bin/python revision/p11_proxy_diagnosticity.py  # ~60 min
.venv/bin/python revision/p11b_analyze_and_figure.py     # seconds
```

Compiled models are cached under `revision/models*/`; delete to force rebuilds.

## The two capabilities the shipped runner lacks

1. **`CompiledSolver.solve(b_ext=...)`** drives Transformer 1 with an arbitrary
   run-time right-hand side, with no recompilation. The runner supplies `b_i` at
   the `rhs_i` token exactly as it already supplies `y_i` at `bck_i` — inside the
   declared host/runner/transformer boundary, not a new privilege.

2. **`RecordingCache`** records, for every attention lookup: the argmax position
   (→ selector hit rate against the compiler-labelled float64 trace), softmax
   one-hotness, output finiteness, and the **true** Shannon entropy computed
   *without* the 10⁻³⁰ regulariser that produced the disowned 10⁻²⁸ nats figures.

## Three findings that change claims in the paper

- **12 of 154 circuits never execute the solve.** Their target node is a fixed
  source node, so the compiler emits a direct source read; the LU machinery is
  built and never exercised. They pass the headline 50 mV metric at 0 mV error.
  154 − 12 = 142, which accounts for the paper's unexplained "142-row file named
  as if 154".
- **The stated cause of the int8 collapse is wrong.** Varying the selector scale
  K over six orders of magnitude (10¹⁰ → 10⁴) leaves the collapse unchanged, and
  K cancels exactly from the closed-form margin condition.
- **90% pruning invariance is entirely structural zeros.** Pruning 90% of all
  weights is bit-identical; pruning 10% of the *nonzero* weights destroys
  execution (η → 1.0).

## Not addressed

P0.8 (citation existence audit — needs the manuscript bibliography, absent from
this repository), the bounded-magnitude positional re-encoding of P1.1/H2a (a
compiler change, not a parameter sweep), P1.2, P1.5, P1.8, and all P2 items. See
the "What we did not do" section of the response for details.
