# Experiment Summary

Total circuits: 154

## Accuracy per tier (DSL evaluator, T per tier)

| Tier | N | Pass | Fail | Pass rate |
|------|---|------|------|-----------|
| Basic | 22 | 19 | 3 | 86.4% |
| Intermediate | 32 | 23 | 9 | 71.9% |
| Hard | 100 | 48 | 52 | 48.0% |
| **Total** | 154 | 90 | 64 | 58.4% |

## MILP universality (E7)

- Unique n_layers values across all circuits: `['5']`
- Unique d_model values across all circuits: `['36']`

## Transformer ≡ DSL (E2 sample)

- Sampled circuits: 10, transformer_matches_dsl=True: 10

## Heat diffusion (E11)

Pass tolerance: `err_vs_jacobi <= tol_V` (V_STEP-compounded quantization floor on multi-node Jacobi).

| Grid | N | T | Pred | Jacobi ref | Analytical | err vs Jacobi | err vs Analytical | tol | Pass |
|------|---|---|------|------------|------------|---------------|-------------------|-----|------|
| 3x3 | 10 | 1000 | 5.2500 | 5.2500 | 5.2500 | 0.000000 | 0.000000 | 0.1 | True |
| 5x5 | 26 | 1300 | 5.2000 | 5.2498 | 5.2500 | 0.049800 | 0.050000 | 0.1 | True |
| 10x10 | 101 | 5050 | 4.6500 | 4.9297 | 4.9309 | 0.279700 | 0.280875 | 0.1 | False |
