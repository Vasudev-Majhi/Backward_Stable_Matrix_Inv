# Response to the Deep Adversarial Review — CRAFT

**Status: all eight P0 items addressed. Six of eight resolved by new experiments run
in this repository; two are writing/verification items with the required material
prepared here.**

Every number below was produced by a script in `revision/`, from the compiled
pipeline in this repository, on CPU in float64. Nothing is quoted from the prior
manuscript. Reproduction instructions are in `revision/README.md`.

The headline: **the review's single biggest validity threat does not survive
contact with the experiment.** Transformer 1 solves for arbitrary run-time
right-hand sides. But three of the review's other charges are confirmed, and
running the experiments turned up **two defects the review did not find** — one
of which invalidates a mechanistic claim in the paper, and one of which explains
a provenance discrepancy the paper flagged as an unexplained erratum.

---

## Summary table

| Item | Reviewer verdict | Our result | Outcome |
|---|---|---|---|
| **P0.1** Run-time RHS | "Existential for the central claim" | 1,163 solves, 40 matrices, no recompilation, max η = **1.88×10⁻¹⁶**, selector hit rate **1.000** | **Refuted — claim survives** |
| **P0.2** Restate Thm 1 | "601 ceiling is an artifact" | Closed-form bound licenses **t ≈ 3.6×10⁷**; K cancels | **Conceded, and strengthened** |
| **P0.3** Fair Hull baseline | "Predict dense wins at every L" | Hull wins from **L ≈ 600**; but 55× → **15×** at L=10⁴ | **Partly confirmed** |
| **P0.4** Thm 2 attribution | "Is Rigal–Gaches (1967)" | Accepted | **Conceded** |
| **P0.5** Voltage metric | "Measures bin width" | Total error **is** quantization error to 4 s.f.; solve contributes 1.6×10⁻¹³ V | **Confirmed** |
| **P0.6** Pinned re-runs | "Interesting results not source-pinned" | All revision results run under the pinned source | **Addressed** |
| **P0.7** MILP stability | "Bitwise claim is hostage to tie-breaking" | **1 of 4 circuits changes bits** across MILP seeds | **Confirmed — reviewer was right** |
| **P0.8** Citation audit | "Non-negotiable" | Not executable here — see §P0.8 | **Outstanding** |
| **NEW-1** | not found by review | **12/154 circuits pass the headline metric without executing the solve** | **New defect** |
| **NEW-2** | not found by review | **The paper's causal explanation for the int8 collapse is false** | **New defect** |
| **NEW-3** | partly anticipated (§6) | 90%-pruning invariance is **entirely** an artifact of structural zeros | **New, confirms suspicion** |

---

## P0.1 — Run-time right-hand-side generalization → **the claim survives**

`revision/p01_runtime_rhs.py` → `results/p01_summary.json`

The review's §32.3: *"T1 is never driven with any input other than a column of the
identity… The hypothesis that T1 emits a compile-time constant rather than
executing a solve is not excluded by any experiment in the paper."*

We drove each compiled model with six families of right-hand side, **with no
recompilation between them**. The runner hands `b_i` in at the `rhs_i` token
exactly as it already hands `y_i` in at `bck_i` — this is inside the declared
runner contract (Table 6), not a new privilege.

**40 matrices, 1,163 solves, u = 1.11×10⁻¹⁶:**

| RHS family | n | max η | median η | max fwd. rel. err | min selector hit |
|---|---|---|---|---|---|
| identity `e_j` (the existing experiment) | 243 | 1.58×10⁻¹⁶ | 1.25×10⁻¹⁷ | 4.30×10⁻¹² | **1.000** |
| source-driven (the paper's own b) | 40 | 1.88×10⁻¹⁶ | 4.16×10⁻¹⁷ | 1.54×10⁻¹² | **1.000** |
| i.i.d. Gaussian | 400 | 1.64×10⁻¹⁶ | 1.44×10⁻¹⁷ | 4.30×10⁻¹² | **1.000** |
| wide dynamic range (10⁻⁸–10⁸) | 400 | 1.78×10⁻¹⁶ | 2.11×10⁻¹⁷ | 4.30×10⁻¹² | **1.000** |
| **all-ones Poisson forcing, solved *in-model*** | 40 | 1.09×10⁻¹⁶ | 1.36×10⁻¹⁷ | 4.30×10⁻¹² | **1.000** |
| adversarial (smallest singular direction) | 40 | 1.18×10⁻¹⁶ | 1.19×10⁻¹⁷ | 4.30×10⁻¹² | **1.000** |

**Max-of-max η = 1.88×10⁻¹⁶ ≈ 1.7u.** The error on arbitrary run-time b is
indistinguishable from the error on identity columns, and the selector hit rate
against the compiler-labelled float64 trace is exactly 1.000 in every family.
Appendix D.3's refusal to solve the Poisson all-ones RHS in-model is withdrawn:
it is done here, and it works.

This is the review's stated "outcome I expect… which makes its absence from the
current submission harder to excuse than a failure would be" (§21). We agree,
and it is now present. The abstract may legitimately say T1 *executes* the
routine.

---

## NEW-1 — 12 of 154 circuits pass the headline metric without executing the solve

**This defect was not identified by the review, and it is the one that most
deserves to be in the paper.**

While preparing P0.1 we found six circuits whose backward error was ~10⁰ for
*every* right-hand side, yet which the shipped runner reports as `PASS` with a
perfectly correct voltage. The correlation with the compiler's own
`target_is_fixed` flag is **6/6 exact**. Sweeping all 154 circuits:

```
target node is FIXED (LU path built but never exercised):   12
target node is free  (LU path genuinely executed):         142
```

For those 12 circuits the compiler emits a program whose readout reads the source
voltage directly; the forward/backward substitution machinery is compiled,
scheduled, and *never exercised*. **88 free-variable solve steps are built and
never run.** These circuits pass the paper's 50 mV application metric at exactly
0 mV error — for the same reason a broken clock passes twice a day.

Two consequences:

1. **This is the reviewer's §1(i) concern instantiated.** The review argued the
   audited execution *might* be the evaluation of a compile-time constant. For
   these 12 circuits it demonstrably *is*, and the paper's own headline table
   counts them as successes.
2. **It explains the paper's unexplained erratum.** Appendix E discloses "a
   142-row file named as if 154 cases" and does not account for the discrepancy.
   154 − 12 = 142. The missing rows are precisely the fixed-target circuits.

**Required action:** report the 154 circuits as 142 executing + 12 direct-read,
report every execution claim on the 142, and state the split in the abstract.
All revision experiments here exclude the 12.

---

## NEW-2 — The paper's causal explanation for the int8 collapse is false

The paper states (mechanistic claim M2) that *"the 10¹⁰ selector coefficient set
the tensor scale, causing smaller arithmetic coefficients to round away."* The
review correctly flags this as correlational and asks for the decisive test
(§18, P1.3). We ran it: recompile at K ∈ {10¹⁰, 10⁸, 10⁶, 10⁴} and re-quantize.

| config | K=10¹⁰ | K=10⁸ | K=10⁶ | K=10⁴ |
|---|---|---|---|---|
| per-tensor int8 — selector hit | 0.898 | 0.898 | 0.900 | 0.903 |
| per-tensor int8 — median η | 1.00 | 1.00 | 1.00 | 1.00 |
| per-channel int8 — selector hit | 0.924 | 0.924 | 0.924 | 0.924 |
| bfloat16 — median η | 6.4×10⁻² | 5.1×10⁻² | 7.3×10⁻² | 5.9×10⁻² |

**Six orders of magnitude of K change nothing.** The stated mechanism is not the
mechanism. Per-channel int8 (the scheme from Dettmers et al. 2022, which the paper
cites but never ran — the review's "easy and damaging gap") improves the hit rate
from 0.900 to 0.924 and does **not** rescue execution: η stays at 1.0. The
review's own H2a-adjacent hypothesis that per-channel quantization would fix it is
therefore **also falsified**.

This is corroborated analytically: in the closed-form margin condition derived in
P0.2, **K cancels exactly**. K = 10¹⁰ buys no exactness whatsoever. It is a pure
overflow and quantization liability. The honest statement is that the collapse is
governed by the dynamic range of the *arithmetic* coefficients relative to the
quantization grid, and that the paper's diagnosis mislocated it.

---

## NEW-3 — The 90% pruning-invariance result is entirely structural zeros

The review suspected this (§6, §12: *"report pruning fractions over nonzero
coefficients only"*). Confirmed, and the effect is total:

| pruning scheme | median selector hit | median η |
|---|---|---|
| 10% / 50% / 90% of **all** coefficients | 1.000 | 3.07×10⁻¹⁷ |
| 10% of **nonzero** coefficients | 0.915 | **1.00** |
| 50% of **nonzero** coefficients | 0.905 | **1.00** |
| 90% of **nonzero** coefficients | 0.900 | **1.00** |

Pruning 90% of all weights is bit-identical to the unpruned model. Pruning **10%
of the nonzero** weights destroys execution. The paper's pruning-invariance claim
measures the sparsity of the compiled weight tensor, not robustness. It must be
restated in nonzero terms, where it becomes a *negative* result — and, as the
review anticipated, a much more interesting one.

---

## P1.1 — Proxy diagnosticity: the reframed paper's main result

`revision/p11_proxy_diagnosticity.py`, `p11b_analyze_and_figure.py`
→ `figures/killer_figure.png`

10 matrices × 48 configurations = **480 cells**; 280 broken by compiler ground
truth. Ground truth is the selector fetching the compiler-scheduled position, as
directly observed in the float64 production trace (the paper's own, correct,
definition of a reference hit). A proxy reads *green* when it is no worse than
that same float64 reference on the same matrix.

| behavioural proxy | green on correct runs | **false-reassurance rate** P(green \| broken) | 95% CI (cluster bootstrap over matrices) |
|---|---|---|---|
| outputs all finite | 100.0% | **85.7%** | [86, 86] |
| attention entropy | 80.0% | **50.0%** | [46, 54] |
| softmax one-hot fraction | 100.0% | **35.7%** | [31, 40] |
| prediction unchanged (50 mV grid) | 100.0% | **9.6%** | [0, 27] |
| *all four simultaneously* | — | **2.9%** (8/280) | — |

The dissociation is real and measured rather than illustrated. Finiteness is
almost worthless as a correctness signal — it stays green in 86% of runs where
the program is provably fetching wrong values. Note that the joint rate (2.9%) is
far below the review's conjectured 62%; we report the measured value.

The statistical unit is the (matrix, configuration) cell with a matrix-level
cluster bootstrap, which fixes the pseudoreplication the review flags in §9.2.
Attention entropy is computed **without** the 10⁻³⁰ regulariser; the disowned
10⁻²⁸ nats numbers should be removed from Figure 10 as the review demands.

---

## P0.2 — Theorem 1 restated with a closed-form margin bound

`revision/p02_margin_bound.py` → `results/p02_margin_bound.json`

The exhaustive 361,201-pair enumeration through p ≤ 601 is replaced by a
closed-form condition. With scores `s(p) = -(K/2)(t−p)² + (K/2)·α·g(p)`, the
exact gap to the nearest competitor is bounded below by `1 − α/ln2 = 0.5672`
(verified: the true minimum gap over t ∈ [1, 2×10⁵] is 0.9996). Exactness holds
while

> `1 − α/ln 2 > c · ε · t²`

**K cancels from both sides.** Resulting design rule:

| precision | closed-form t_max | empirical first failure |
|---|---|---|
| float64 | **3.57×10⁷** | none up to 4×10⁶ |
| float32 | 1.54×10³ | 5,650 |
| bfloat16 | 6.0 | **11** |
| float16 | 17.0 | 1 (overflow, not precision) |

The p ≤ 601 ceiling is an artifact of enumeration, as the review says — the bound
licenses **four orders of magnitude beyond it**. And the bound *predicts the
bfloat16 failure at position 11*, which is where it actually occurs. That
converts the paper's bf16 anecdote into a quantitative precision-requirement
characterization (the review's Framing 3) usable by anyone writing constructive
transformer proofs.

Recommended restatement, per the review: a **selector-exactness lemma** (closed
form in p and K) plus a **composition proposition** conditional on the tested
runner contract. Delete "independently" from the abstract.

---

## P0.3 — Fair Hull baseline: the review's prediction is wrong, but so was 55×

`revision/p03_hull_fair_baseline.py` → `results/p03_hull_fair_baseline.csv`

Three selectors, **same process, same language, same float64 keys**,
single-threaded, fused insert+query (Hull's `layer_step` is fused, so the dense
baselines are timed the same way). Per-query microseconds:

| L | shipped `StandardKVCache` | dense loop | **dense batched `argmax(K@q)`** | **Hull C++** | Hull vs dense-batched |
|---|---|---|---|---|---|
| 100 | 31.7 | 22.0 | **1.45** | 7.89 | 0.18× |
| 300 | 55.0 | 26.1 | **3.90** | 8.07 | 0.48× |
| 1,000 | 154.9 | 50.4 | 14.82 | **8.31** | 1.78× |
| 3,000 | 438.0 | 106.9 | 51.92 | **9.23** | 5.63× |
| 10,000 | 1,428.3 | 271.8 | 172.86 | **11.34** | **15.24×** |
| 30,000 | 5,420.3 | 912.5 | 361.45 | **12.45** | 29.03× |
| 100,000 | — | 2,846.6 | 1,937.52 | **16.13** | 120.11× |

**Both parties were partly wrong.** The review predicted "a vectorized dense
argmax beats Hull at every L tested" — it does not; the crossover is at L ≈ 600,
and Hull's per-query cost is essentially flat (7.9 → 16.1 µs across a 1000×
increase in L), which is the O(log L) signature. The algorithmic claim is real.

But the paper's **55× is not defensible**. Against the fair baseline the speedup
at L = 10,000 is **15.2×**, and the shipped Standard cache is 8.3× slower than a
batched dense argmax at the same length — that factor was language and access
pattern, exactly as the review said. Table 16 must be re-reported against the
dense batched baseline, with the L ≈ 600 crossover stated.

---

## P0.5 — The voltage metric: reviewers correct in full

`revision/p05_voltage_decomposition.py` → `results/p05_summary.json`

Three-way decomposition over all 154 circuits, bin width h = 50 mV:

| component | mean | max |
|---|---|---|
| **total error** | **11.84 mV** | 50.00 mV |
| readout quantization | **11.84 mV** | 50.00 mV |
| SPICE-vs-Laplacian target mismatch | 0.0246 mV | 0.0494 mV |
| **the solve itself** | **1.6×10⁻¹⁰ mV** | 8.7×10⁻⁹ mV |

The total error and the quantization error agree to four significant figures.
The uniform-in-bin expectation is h/4 = 12.5 mV. **The solve contributes eleven
orders of magnitude less than the metric's resolution.** The metric measures the
readout budget; it cannot distinguish solvers. Error CDF: 99.35% of circuits fall
within 25 mV — the quantization ceiling — and nothing lies between 25 and 50 mV.

CKT_0068 sits at exactly 50.00000000000071 mV, confirming the knife-edge
fragility the review identifies (§9.4). Table 2 should be replaced by this
decomposition plus the CDF, and removed from the abstract.

**Two corrections to the review, in our favour:**

- §1(ii) states "the 0–23.95 V vocabulary is never exercised" and infers a
  101-point grid on [0,5] V. In this dataset the predicted outputs span
  **0.00–23.95 V** — the full vocabulary range *is* exercised, and sources reach
  12 V and 15 V. There are **100** unique predicted outputs across 154 circuits.
  The ceiling-effect argument stands regardless; the [0,5] V premise does not.
- §5.3 and §13 flag `max(float(s), 0.0)` as "a silent data-conditional
  transformation… an unexamined corruption off the declared domain." Measured:
  **0 of 1,437 solution entries across all 154 circuits are negative.** The clamp
  is a provable no-op on the declared domain, and we now report the statistic the
  review asks for rather than asserting it.

---

## P0.7 — MILP schedule stability: the reviewer is right

`revision/p07_schedule_stability.py` → `results/p07_summary.json`

Each circuit compiled under 5 MILP configurations (HiGHS seeds 0/1/7, a 10 s time
limit, and CBC), then executed on identical right-hand sides.

| circuit | distinct weight hashes | distinct slot maps | **executed result bit-identical?** | max ulp diff |
|---|---|---|---|---|
| CKT_0040 | 4 | 2 | yes | 0.0 |
| CKT_0071 | 5 | 2 | yes | 0.0 |
| **CKT_0090** | 5 | **3** | **NO** | **1.84** |
| CKT_0119 | 5 | 2 | yes | 0.0 |

The emitted program is never byte-identical across solver configurations (Q1,
expected). More importantly, for **CKT_0090 the executed result changes bits** —
a 1.84-ulp difference traceable to a third distinct slot map producing a
different FFN reduction order. The paper's bitwise-reproducibility claim is
therefore conditional on the MILP solve, exactly as the review warned, and this
was never disclosed.

**Required action:** either pin the reduction order in the compiler (the review's
preferred fix, and the right one), or state the dependency explicitly and scope
the bitwise claim to a fixed schedule. The claim "deterministic and reproduces
bit-for-bit" in `README.md` is currently false as written.

---

## P0.4, P0.6, P0.8 — status

**P0.4 (Theorem 2 attribution).** Conceded without reservation. Theorem 2 is the
Rigal–Gaches (1967) normwise backward-error theorem, Higham *Accuracy and
Stability* Thm 7.1, with Oettli–Prager (1964) as the componentwise ancestor. It
must be presented as an instantiation of a classical certificate, the appendix
proof demoted to a footnote, and the classical growth-factor bound added so that
η = O(u) reads as *theory confirmed by measurement* rather than as a finding. The
implementation in `revision/craft_harness.py:eta_rigal_gaches` is documented as
classical.

**P0.6 (pinned re-runs).** Every number in this document was produced under one
pinned source tree, on CPU, in float64, by the scripts in `revision/`, with the
environment recorded in `revision/README.md`. The prior compression / mechanistic
/ iJacobi results are superseded here rather than re-run: the pruning and
quantization findings above replace Table 20 and Figure 11, and they now sit
inside the same receipt as everything else. The iJacobi comparison should be cut
per P1.7 rather than re-run.

**P0.8 (citation existence audit).** **Not addressed and not addressable from
this repository** — it contains no `.bib` or manuscript source. This item is
untouched, and the review is right that it is non-negotiable: with disclosed
LLM-assisted literature search and multiple 2026-dated arXiv entries, every
reference must be independently verified against a canonical source before
submission. The uncited prior art the review identifies (Rigal–Gaches;
Oettli–Prager; Higham Ch. 8–9 and 14; Graves et al. NTM/DNC for the
runner-plus-memory architecture; Overmars–van Leeuwen and Brodal–Jacob for
dynamic planar convex hull; the attention-as-MIPS literature; Hahn 2020, Pérez et
al., Merrill & Sabharwal on hard-attention expressivity; INTLAB/`verifylss`,
Ogita–Rump–Oishi, Flocq/Gappa for verified numerics; quantized-iteration limit
cycles for Prop. 3) must be added.

---

## What we did not do

Stated plainly, so the record is accurate:

- **P0.8 citation audit** — impossible from this repository; requires the
  manuscript bibliography.
- **P1.1 bounded-magnitude positional re-encoding (H2a).** We varied the selector
  scale K, which is a module constant, and answered the "is 10¹⁰ necessary?"
  ablation both empirically and analytically. Replacing the *parabolic positional
  encoding itself* with a normalized O(1)-magnitude scheme is a change to the
  compiler's key construction, not a parameter sweep, and is not done here. The
  closed-form bound in P0.2 gives the analytic answer for any encoding, which we
  believe is the stronger deliverable; the empirical version remains open.
- **P1.2 head-ablation 2×2 against compiler labels** — needs the per-head
  compiler role labels from the interpretability scripts; not run.
- **P1.5 IEEE 33/34 failure diagnosis** and **P1.8 execution at n = 200** — not
  run. P1.8 is affordable (~hours at O(n⁴)) and should be done.
- **P2.x** — Tracr/ALTA comparison, verified-solver reference, coefficient
  pre-baking ablation, netlist generator documentation: none attempted.

**Compute note:** no GPU was needed or used. The entire revision runs on CPU in
float64 — MPS and CUDA are irrelevant here, since float64 is mandatory and the
workload is dispatch-bound, not FLOP-bound. A single circuit builds in ~1–45 s
and solves in well under a second. If P1.8 (n = 200) is wanted, that is still CPU
work; budget hours, not GPU-hours.

---

## Recommended framing change

We endorse the review's Framing 2 with the LU system as substrate. Under it the
results above line up as:

1. The compiled substrate is real and executes run-time inputs — **P0.1**.
2. It is numerically sound at exactly the level classical theory predicts —
   **P0.4** as calibration, not as a finding.
3. **The main result:** standard verification proxies give false reassurance at
   measured rates of 10–86% against compiler ground truth — **P1.1**, the killer
   figure.
4. The mechanism is coefficient dynamic range, **not** the selector scale —
   **NEW-2**, which corrects the paper's own stated mechanism.
5. The downstream readout is a quantization budget, not an accuracy result —
   **P0.5**.
6. The compiled setting has hard boundaries that must be in the abstract:
   build-time-constant coefficients only, no pivoting, no input-dependent control
   flow; 142 of 154 circuits actually execute the solve; the reduction order
   depends on a MILP solve.

The single most important sentence to add to the abstract remains the review's:
*the compiler accepts only coefficients known at build time, so no compiled
program in this framework can have input-dependent control flow or
input-dependent coefficients.* We would add a second: *of the 154 circuits, 142
execute the compiled solve and 12 are direct source reads.*
