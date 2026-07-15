# CRAFT — Matrix Inversion Inside a Transformer via Backward-Stable LU Solves

Reference code, data, and compiled artifacts for the paper
**"Matrix Inversion Inside a Transformer via Backward-Stable LU Solves"** (CAISc 2026).

CRAFT **compiles** (does not train) the triangular solves of matrix inversion —
forward/back substitution — into a transformer's forward pass, at **float64 machine
precision**, for symmetric positive-definite (SPD) Laplacian submatrices. The `L·U`
factorisation itself runs once at build time in NumPy and is baked into the token
embeddings; the transformer executes the remaining ~2/3 of the inversion FLOPs.

> **Scope, honestly.** CRAFT is an **expressivity and interpretability ground-truth**
> result, **not a practical solver** — end-to-end it is ~10⁷× slower than
> `numpy.linalg.solve`. Its value is (a) exact, provably-correct weights as ground truth
> for mechanistic interpretability, (b) a proof that transformers can represent exact
> float64 linear algebra, and (c) a characterisation of when compiled *iterative* solvers
> fail (Proposition 3). See the paper for the full framing.

Weights are **constructed analytically** (three primitives — `fetch`/`reglu`/`persist` —
placed by an MILP scheduler), so results are **deterministic and reproduce bit-for-bit**.
**float64 is mandatory** (the `HARD_K = 1e10` attention scale overflows in float32).

---

## Repository layout

| Path | What it is |
|---|---|
| `transformer-vm/` | The CRAFT compiler library: `fetch`/`reglu`/`persist` primitives (`transformer_vm/graph`), the MILP scheduler (`transformer_vm/scheduler`), the DSL→PyTorch weight builder (`transformer_vm/model`), and the **Hull KV cache** C++ extension (`transformer_vm/attention`). |
| `lu_pipeline/` | The **LU-direct pipeline** — build (`build_lu_direct.py`), run (`runner_lu_direct.py`, `lu_direct_run_all.py`), the Doolittle factoriser (`lu_factor.py`), plus `cross_domain_run.py`, `compression_sweep.py`, `tolerance_sweep.py`, tests, and the 154 compiled model sidecars (`model_CKT_*_lu_direct_v2.bin.slots.json`). |
| `dataset/` | `circuit_dataset_rv.jsonl` — the **154 in-house SPICE resistor circuits** (N=3–50), one JSON object per line (netlist, target node, ground-truth voltage). |
| `craft/cadj/` | Iterative baselines + the **Transformer-2 readout**: `idea2/` (iJacobi + the sensitivity-matrix readout transformer), `idea3/sidecars/` (per-circuit sidecars + compiled `lu_model` binaries). |
| `craft/rbsor/` | Compiled **Red-Black SOR** iterative baseline. |
| `craft/tmlr2.0/experiments/` | Paper experiment scripts: `E0_leaderboard.py` (attribution-method leaderboard), `E1_backward_error.py`, `N1_symbolic_trace.py`, `N2_lapack_compare.py`, `N4_fem_*.py`, `N5_acdc*.py`, `T2_vs_numpy_dot.py`, `T3_milp_vs_greedy.py`, `T5_tolerance_sweep.py`, `T9_hardk_sweep.py`. |
| `craft/tmlr2.0/results/` | All result CSVs (LU sweeps, `compression_154.csv`, `kappa_154.csv`, backward-error, cross-domain summaries). |
| `craft/experiments/` | V_step / scaling / failure-mode experiments (`e12_vstep.py`, `e13_tscaling.py`, `e14_failure_modes.py`) and shared helpers. |
| `craft/benchmarks/`, `ieee_benchmarks/`, `benchmark_datasets/` | Cross-domain benchmark **harness + data** (SuiteSparse SPD, social/biological graphs, 2D resistor lattices, FEM Poisson, pandapower/raw IEEE bus systems). |
| `matrix_inversion/inversion2/` | Large-N (N=200) Hull-KV rerun code + results. |
| `paper_experiments/`, `compression/` | Mechanistic-interpretability code: per-head ablation, residual probing, attention-argmax (`interp_t1.py`, `interp_t2.py`), and the compression sweep (pruning vs. quantization). |

---

## Installation

Requires **Python 3.11**. Reference environment: PyTorch 2.6 + CUDA 12.4, NumPy (MKL BLAS).

```bash
pip install -r requirements.txt
```

You do **not** need to install the `transformer-vm` library separately — the scripts add it
to the path automatically (via `_bootstrap.py`, which finds `transformer-vm/` at the repo
root). If you prefer an explicit install, `pip install -e ./transformer-vm` also works.

Point the code at the bundled dataset once (the entry points also default to
`dataset/circuit_dataset_rv.jsonl` at the repo root):

```bash
export CRAFT_DATASET="$PWD/dataset/circuit_dataset_rv.jsonl"   # Linux/macOS
# setx CRAFT_DATASET "%CD%\dataset\circuit_dataset_rv.jsonl"   # Windows
```

### Hull KV cache

The Hull KV cache (which lowers attention cost from `O(seq²)` to `O(seq·log seq)` and lets
inference scale to N≈200) is a **C++17 extension** JIT-compiled at import time via
`torch.utils.cpp_extension`. It needs a C++17 compiler on PATH **in addition** to the pip
packages `pybind11` and `ninja`:

- **Linux:** `g++` or `clang` (the paper's build used `clang` + `lld`)
- **macOS:** `clang` (`brew install llvm lld`)
- **Windows:** MSVC Build Tools — **or** skip the C++ build entirely: a pure-Python
  reference, `brute_hull_cache.py`, produces bit-identical results (just slower). The
  standard `O(seq²)` softmax cache (`transformer_vm/attention/standard_cache.py`) also
  needs no compiler.

---

## Quickstart

Run the full compiled-LU sweep over the 154 circuits (build → run → check vs. NumPy/LAPACK):

```bash
cd lu_pipeline
python lu_direct_run_all.py          # writes results/results_main_lu_direct.csv
```

Cross-domain benchmarks (SuiteSparse, graphs, lattices, IEEE, …):

```bash
cd lu_pipeline
python cross_domain_run.py --all                 # all 30 datasets
python cross_domain_run.py --case karate         # one dataset
python cross_domain_run.py --case karate --hull  # use the Hull KV cache
```

Because the weights are exact and every dimension has a known job, the same circuits serve
as a float64 ground truth for interpretability:

```bash
cd lu_pipeline && python compression_sweep.py        # pruning (bit-identical @90%) vs int quant (collapses)
```

The per-head ablation / residual-probing / attention-argmax experiments live in
`paper_experiments/interp_t1.py` (Transformer 1) and `interp_t2.py` (the readout transformer).
Per their docstrings, run them with `matrix_inversion/inversion2/` on the path so the
`runner_lu` imports resolve, e.g. `cd matrix_inversion/inversion2 && python <path>/interp_t1.py`.

---

## Compiled model weights

The full compiled transformer weights are **deterministically regenerable** from the build
scripts (`lu_pipeline/build_lu_direct.py`), which is the point of a *constructed* model — nothing
is trained. To keep the repository small we ship:

- **All 154** `*.bin.slots.json` sidecars (the per-circuit compiled-model metadata), and
- **4 representative** raw `.bin` weight tensors (circuits CKT_0001/0034/0098/0143, ~2.3 MB
  total) under `craft/cadj/idea3/sidecars/`, used by the interpretability experiments.

To materialise any other circuit's weights, run the build script for that circuit ID.

---

## Notes & limitations

- **Determinism:** single-threaded float64; runs reproduce bit-for-bit. There is no training,
  so there are no seeds, error bars, or stochasticity to report.
- **float64 only:** `HARD_K = 1e10` overflows in float32; do not downcast.
- **Scale:** inference is `O(N³)`; practical reach is `N ≲ 200` with the Hull KV cache. Model
  width grows as `d_model ≈ 12N`; per-circuit recompilation is required when `A_FF` changes.
- **`compression/analyze_sweep.py --pull`** references an (intentionally omitted) server helper;
  the default local mode works from the shipped result files.

## Citation

Please cite the CAISc 2026 paper. **License:** Apache-2.0 — see [`LICENSE`](LICENSE)
(the bundled `transformer-vm` compiler library is Apache-2.0 as well).
