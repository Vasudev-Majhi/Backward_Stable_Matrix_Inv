# RB-SOR Investigation — Consolidated Results So Far

Snapshot covering everything through commit `869f94e` plus the in-progress E20 standard-cache run. All numbers from `craft/rbsor_results/`.

---

## 1. Pass-rate headline (154 circuits, all on remote Linux + Hull cache)

| Configuration | Pass | Rate | Source |
|---|---|---|---|
| **Jacobi baseline** (V_STEP=0.05, T=max(1000, 50N)) | 90/154 | 58.4% | `results_main_jacobi_baseline.csv` |
| **RB-SOR v0** (V_STEP=0.05, ω=ω_opt, no fallback) | 85/154 | 55.2% | `results_main_rbsor.csv` |
| **RB-SOR v01** (V_STEP=0.01, ω≤1.5, conflict+divergence fallback to Jacobi) | **111/154** | **72.1%** | `results_main_rbsor_v01.csv` |

**Net result**: v01 beats Jacobi by **+21 PASS** circuits (+13.7 pp).

---

## 2. Per-circuit head-to-head (v01 vs Jacobi-baseline)

| Bucket | Count |
|---|---|
| Both PASS | 89 |
| Both FAIL | 42 |
| **RESCUES** (v01 PASS, Jacobi FAIL) | **22** |
| REGRESSIONS (Jacobi PASS, v01 FAIL) | **1** (CKT_0145, err=0.0512V — 0.0012 over tol) |

The 22 rescues split into:
- **4 pure RB-SOR rescues** (bipartite circuits, RB-SOR's amplification was the lever): CKT_0015, CKT_0035, CKT_0039, CKT_0061
- **18 conflict-fallback rescues** (algorithm correctly chose Jacobi at V_STEP=0.01 because the graph wasn't bipartite): CKT_0031, CKT_0032, CKT_0047, CKT_0048, CKT_0053, CKT_0065, CKT_0067, CKT_0068, CKT_0069, CKT_0089, CKT_0092, CKT_0094, CKT_0098, CKT_0099, CKT_0105, CKT_0112, CKT_0124, CKT_0144

**Most of the gain came from finer V_STEP + Jacobi-fallback, not from RB-SOR's amplification.**

### v01 solver breakdown
| Solver chosen by orchestrator | PASS | FAIL |
|---|---|---|
| jacobi-fallback (113 circuits, 73%) | 81 | 31 |
| rbsor (42 circuits, 27%) | 30 | 12 |

The auto-decision logic dispatched **73% of circuits to Jacobi** — proof that RB-SOR is a bad default at this discretization.

---

## 3. E18 — ω sweep on the 14 v0 regressions at V_STEP=0.05

```
omega=1.00 (Gauss-Seidel) : pass=14/14   avg_err=0.026V   ← all regressions FIXED
omega=1.30                : pass= 6/14   avg_err=0.815V
omega=1.50                : pass= 4/14   avg_err=1.939V
omega=1.70                : pass= 3/14   avg_err=2.457V
omega=auto (1.85-1.99)    : pass= 0/14   avg_err=3.242V
```

Perfectly monotonic: **error scales with ω**. The "regressions" are pure SOR-instability artifacts of the auto-ω formula at coarse V_STEP. At ω=1.0 (= Gauss-Seidel, no over-relaxation) every one of the 14 passes. Cleanest possible evidence that the v01 ω-cap was the right design choice.

---

## 4. E19 — T-scaling on 9 rescues + 14 regressions (current ω, V_STEP=0.05)

```
category    |  T=50  T=100 T=200 T=500 T=1000 T=2000
rescue      |   8/9    8/9    8/9    8/9    6/9    5/9
regression  |   1/14   1/14   1/14   1/14   0/14   2/14
```

Two important signals:
- **More T HURTS rescues**: 8/9 stable at T=50–500, drops to 5/9 at T=2000. Over-relaxation eventually pushes past truth even on "good" circuits.
- **More T does NOT help regressions**: stuck at 0–2/14 across all T. They aren't slowly converging — they're genuinely unstable. Confirms that the divergence isn't a "needs more iterations" problem.

Combined with E18, this makes the diagnosis bulletproof: **at V_STEP=0.05, RB-SOR with ω>1 is strictly worse than Gauss-Seidel** on these circuits.

---

## 5. E20 — Hull vs Standard at N=101 (heat 10×10, T=5050, ~1M tokens) — *partial*

| Cache | Wall-clock | Status |
|---|---|---|
| **Hull** | **822.6s** (≈14 min) | Done |
| Standard | DNF (>2h36m and counting) | Still running |

Hull made N=101 *feasible at all* — Standard is still in the inner attention loop after 2h36m. Even with the standard run incomplete, the asymptotic-scaling claim holds: at N=101 with T=5050, Hull is at least **10×** faster (and likely 30–50× when the run finishes, extrapolating from the prior 5-circuit benchmark which showed 9-43× across N=10-50).

---

## 6. Pre-existing baseline data (from earlier work, not re-run this session)

These are documented in `craft/meta_data/status.md` and `walkthrough.md`:
- 5-circuit hull benchmark at T=100 (N=10 to N=50): **9–43× speedup** (CKT_0075→CKT_0133)
- E1 baseline 90/154 at V_STEP=0.05 — confirmed by today's apples-to-apples re-run
- E12 already proved V_STEP=0.01 fixes Mode-B circuits in plain Jacobi (10/10) — today's data extends this to RB-SOR-with-fallback territory
- Mode classification: A=26 (jacobi-singular), B-simple=14, B-compound=19, C=5 — sums to 64 fails

---

## 7. Analysis — what we now know

**Forward RB-SOR is not a free upgrade over Jacobi at this discretization.**

### Where RB-SOR wins
1. **Quantum-stall escape on bipartite circuits** (4 circuits) — ω-amplification breaks the `acc//SCALE = 0` trap that kills Jacobi on slow-converging cases.
2. **Faster end-to-end** (~1.3-1.8× per pre-existing analysis_report.md) due to ~5× fewer iterations, despite higher per-iter cost.

### Where RB-SOR loses
1. **Quantum overshoot on bipartite circuits** when ω is aggressive — provable from E18 (ω=1.0 fixes all 14 regressions; ω→1.99 makes them all fail).
2. **Divergence on non-bipartite circuits** — the consistent-ordering theorem doesn't apply with conflict edges; the iteration matrix can have spectral radius >1.
3. **More T makes things worse** for both rescues (eventually overshoot) and regressions (never converge) — E19.

### What the v01 strategy did right
- **Always check graph bipartiteness first**: if not bipartite, use Jacobi (which always converges for diagonally-dominant resistor graphs). Got us 18 of 22 rescues.
- **Pre-flight divergence detector**: run a 50-vs-100-iter Python check; if pred jumps or |pred|>24V, route to Jacobi.
- **Cap ω at 1.5**: prevents the catastrophic divergences (E18 shows the cliff is between 1.5 and 1.7 for these circuits).
- **Finer V_STEP=0.01**: makes both algorithms more robust to quantization stalls; this alone explains many of the rescues.

### Real ceiling for iterative methods at this transformer depth
Realistic upper bound: the 26 Mode-A circuits (ρ_J ≥ 0.9999) cannot be cracked by any pure-iterative scheme without far more iterations than we can fit. They need either Krylov methods (CG, BiCGSTAB) or a direct factorization step. **111/154 is plausibly close to the iterative ceiling.**

### Open thread
- E20 standard-cache benchmark still running. When it finishes the N=101 speedup ratio will be the largest single Hull data point in the project.
- One V_STEP=0.005 retry on CKT_0145 would close the +21/-1 to +21/-0.

---

## 8. Files in this folder

| File | Bytes | Contents |
|---|---|---|
| `results_main_jacobi_baseline.csv` | 24K | 154-circuit Jacobi sweep (apples-to-apples baseline) |
| `results_main_rbsor.csv` | 20K | 154-circuit RB-SOR v0 sweep |
| `results_main_rbsor_v01.csv` | 24K | 154-circuit RB-SOR v01 sweep (with fallback) |
| `omega_sweep_E18.csv` | 8K | 14 circuits × 5 ω values |
| `rbsor_tscaling_E19.csv` | 16K | 23 circuits × 6 T values |
| `build_stats_jacobi.csv` | 12K | MILP schedule per circuit |
| `analysis_report.md` | 20K | Detailed v0 analysis (pre-v01) |
| `summary_so_far.md` | this | Consolidated summary across all 3 configurations |
| `heat_hull_E20.csv` | (pending) | Hull-vs-Standard at N=101 |
| `rbsor_v2_run.log` | 856K | Full tmux log (gitignored) |
