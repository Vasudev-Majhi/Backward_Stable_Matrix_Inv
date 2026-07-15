# tmlr2.0 results — Layout

Per-experiment folders. Each `Nk/` corresponds to one of the new experiments
in the plan; `_shared/` holds CSVs that were already on the server before
the plan started and that multiple sections cite.

```
results/
├── N1/   Symbolic execution traces (CKT_0001, CKT_0067)
│   ├── N1_trace_CKT_0001.json   single-column LU inversion trace per token
│   └── N1_trace_CKT_0067.json
├── N2/   LAPACK bit-exact head-to-head
│   └── N2_lapack_bitexact.csv   CRAFT vs scalar Doolittle vs scipy at N=5..20, kappa=1..1000
├── N3/   Trained-baseline precision-vs-steps curve (LONG; in progress)
│   ├── prior_run1_train.log     20k-step run that produced baseline_comparison.csv
│   ├── N3_train.log             current 200k-step run (in progress on server)
│   ├── curve.jsonl              one row per val checkpoint (pulled when ready)
│   └── N3_precision_curve.csv   final per-step precision curve (after N3_eval_sweep.py)
├── N4/   FEM Poisson on a 2D unit square
│   └── N4_fem_poisson.csv       CRAFT vs scipy.sparse.spsolve at mesh sizes 4..16
├── N5/   Adversarial ACDC per-head ablation
│   ├── N5_acdc_summary.csv      one row per circuit: critical heads / total
│   ├── N5_acdc_CKT_0001.json    per-head edge list (layer, head, err, kept)
│   ├── N5_acdc_CKT_0015.json
│   └── N5_acdc_CKT_0067.json
├── N6_iJacobi/   iJacobi 128/154 verification (already done on server)
│   ├── results_main_ijacob.csv  128/154 PASS at V_STEP=0.05, T=1000
│   └── results_ijacob_hull.csv  124/154 with Hull KV cache
└── _shared/
    ├── CRAFT_main_experiments/  154-circuit pipeline, scaling, compression,
    │                            trained-baseline single-snapshot, kappa
    └── cross_domain/            34 CSVs: IEEE, SuiteSparse, social, lattices
```

## Verified pass rates (use these numbers in the paper, not memory)

| Method | Pass@50mV | File |
|---|---|---|
| CRAFT idea2 pipeline | **150/154** | `_shared/CRAFT_main_experiments/results_idea2.csv` |
| CRAFT LU-direct | **150/154** | `_shared/CRAFT_main_experiments/results_main_lu_direct_v2.csv` |
| iJacobi (T=1000, V_STEP=0.05) | **128/154** | `N6_iJacobi/results_main_ijacob.csv` |
| iJacobi + Hull KV | 124/154 | `N6_iJacobi/results_ijacob_hull.csv` |
| Trained baseline (6.3M params, 20k steps) | **7/142** | `_shared/CRAFT_main_experiments/baseline_comparison.csv` |

## Pulling results from server

After all tmux sessions finish, run:
```
python _pull_results.py
```
to grab anything not already pulled and route it to the right subfolder.
