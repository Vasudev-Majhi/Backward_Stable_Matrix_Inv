# LU Transformer Pipeline — Visual Overview

---

## PHASE 1 · BUILD THE MODEL *(done once per circuit)*

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐           │
│   │                 │       │                 │       │                 │           │
│   │  SPICE Netlist  │ ───▶  │  Conductance    │ ───▶  │   Partition     │           │
│   │                 │       │  Matrix  A      │       │                 │           │
│   │  R1=68kΩ        │       │                 │       │  A_FF  (free×   │           │
│   │  R2=22kΩ        │       │  A[i,i] = Σg    │       │        free)    │           │
│   │  V1=12V         │       │  A[i,j] = -g    │       │  A_FP  (free×   │           │
│   │                 │       │                 │       │        fixed)   │           │
│   └─────────────────┘       └─────────────────┘       └────────┬────────┘           │
│         PARSE                   NUMPY (build)                   │                   │
│                                                                  │                   │
│                          ┌───────────────────────────────────────┘                   │
│                          │                                                            │
│                          ▼                                                            │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │                   TRANSFORMER 1 — LU INVERSION  (inversion2/)               │   │
│   │                                                                              │   │
│   │  Step A  ┌────────────────────────┐                                         │   │
│   │  NUMPY   │  Doolittle LU          │   A_FF = L × U                          │   │
│   │  (CPU)   │  Factorization         │   L = unit lower-triangular             │   │
│   │          │                        │   U = upper-triangular                  │   │
│   │          └────────────┬───────────┘   computed once, no ML                 │   │
│   │                       │                                                     │   │
│   │  Step B  ┌────────────▼───────────┐                                         │   │
│   │  DSL     │  Computation Graph     │   L,U values baked as constants         │   │
│   │          │                        │   into per-token embeddings             │   │
│   │          │  fwd_0..fwd_{n-1}  ──▶ │   fwd_i  encodes row i of L            │   │
│   │          │  bck_{n-1}..bck_0  ──▶ │   bck_i  encodes row i of U            │   │
│   │          └────────────┬───────────┘                                         │   │
│   │                       │                                                     │   │
│   │  Step C  ┌────────────▼───────────┐                                         │   │
│   │  MILP    │  Scheduling            │   finds minimum d_model                 │   │
│   │          │                        │   that fits all operations              │   │
│   │          │  d_model = 46          │   3 layers, no gradient descent         │   │
│   │          │  n_layers = 3          │   weights derived analytically          │   │
│   │          └────────────┬───────────┘                                         │   │
│   │                       │                                                     │   │
│   │  Step D  ┌────────────▼───────────┐                                         │   │
│   │  OUTPUT  │  model_<hash>_lu.bin   │   transformer weights saved to disk     │   │
│   │          │  + .slots.json         │   sidecar records which residual        │   │
│   │          │                        │   slots hold x_value, b_value, x_new   │   │
│   │          └────────────────────────┘                                         │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 2 · RUN THE LU TRANSFORMER *(build time — produces A_FF⁻¹)*

```
  One column of the identity matrix e_j  (repeated n_free times)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                   TOKEN SEQUENCE   (3·n_free + 2  tokens per column)                │
│                                                                                     │
│  pos 0        pos 1..n            pos n+1..2n          pos 2n+1..3n      pos 3n+1  │
│  ┌────────┐   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   ┌──────┐  │
│  │ start  │──▶│  init_0      │──▶ │  fwd_0       │──▶ │  bck_{n-1}  │──▶│ halt │  │
│  └────────┘   │  init_1      │    │  fwd_1       │    │  bck_{n-2}  │   └──────┘  │
│               │   ...        │    │   ...        │    │   ...       │             │
│               │  init_{n-1}  │    │  fwd_{n-1}   │    │  bck_0      │             │
│               └──────────────┘    └──────────────┘    └──────────────┘             │
│               seed KV cache       FORWARD SUB          BACK SUB                    │
│               with position info  L · y = e_j          U · x = y                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                        │
       ─────────────────────────────────────────────────
       What happens INSIDE the transformer at each token
       ─────────────────────────────────────────────────
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
  ┌───────────────────┐     ┌───────────────────┐
  │    fwd_i token    │     │    bck_i token     │
  │                   │     │                   │
  │ Runner injects:   │     │ Runner injects:    │
  │   b[i] → slot    │     │   y[i] → slot     │
  │                   │     │                   │
  │ Attention reads:  │     │ Attention reads:   │
  │   y_0..y_{i-1}   │     │   x_{i+1}..x_{n-1}│
  │   from KV cache  │     │   from KV cache   │
  │                   │     │                   │
  │ FFN computes:     │     │ FFN computes:      │
  │   y_i = b[i]     │     │   x_i = (y[i]     │
  │   - Σ L[i,j]·y_j │     │   - Σ U[i,j]·x_j) │
  │   (L baked in)   │     │   / U[i,i]         │
  │                   │     │   (U baked in)    │
  │ Runner patches:   │     │ Runner patches:    │
  │   KV cache ←y_i  │     │   KV cache ← x_i  │
  └─────────┬─────────┘     └─────────┬──────────┘
            │                         │
            └────────────┬────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   column j of       │
              │     A_FF⁻¹          │
              │   (x_0..x_{n-1})    │
              └─────────────────────┘

  Repeat for j = 0..n_free-1  →  full  A_FF⁻¹  assembled
```

---

## PHASE 3 · COMPUTE S AND BUILD READOUT MODEL *(build time — pure Python + Transformer 2)*

```
        A_FF⁻¹                   A_FP
   (from Transformer 1)     (from conductance matrix)
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  S = -A_FF⁻¹    │    plain matrix multiply
              │      @ A_FP     │    no linalg.inv or linalg.solve
              │                 │    shape: n_free × n_fixed
              └────────┬────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              TRANSFORMER 2 — READOUT MODEL  (cadj/)                     │
│                                                                          │
│  S values are INTEGER-QUANTIZED and baked into the readout token:        │
│                                                                          │
│    sens_coef[k] = round( max(S[target, k], 0) × 10000 )                 │
│                                                                          │
│  e.g. S=0.7556  →  sens_coef = 7556  (frozen into embedding weights)   │
│                                                                          │
│  MILP builds weights analytically — no gradient descent.                │
│  d_model=48, n_layers=4, vocab=484, ~100,608 params per circuit.        │
│                                                                          │
│  Saved: model_CKT_XXXX_idea2.bin                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 4 · INFERENCE *(runs on every query — Transformer 2 only)*

```
  Source voltages  (e.g. V1=0V, V2=12V)
           │
           ▼
  Quantize:  v_k = round(volts × 10000 / 500)
             12V  →  token "v_240"
              0V  →  token "v_0"
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│              TOKEN SEQUENCE  (2N + 4 tokens)                     │
│                                                                  │
│  start                                                           │
│  skip, v_0      ← node 0 (GND = 0V)                             │
│  skip, v_240    ← node 1 (12V)                                   │
│  skip, v_0      ← node 2 (unknown, placeholder)                 │
│  readout        ← sens_coef [0, 7556] baked into this token     │
│  <PRED>         ← model scores all v_k tokens, picks nearest    │
│  halt                                                            │
│                                                                  │
│  Readout token attention:                                        │
│    fetches v_0 and v_240 from KV cache                          │
│    FFN computes:  v_out = (0×0 + 7556×240) / 10000 = 181.3     │
│    → 181.3 × 0.05V/step = 9.065V... (simplified example)       │
│                                                                  │
│  <PRED> picks closest v_k token  →  predicted voltage           │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
  Output:  pred_V  (e.g. 2.9500V)
  ~10–130ms per circuit
```

---

## COMPLETE PIPELINE — ONE VIEW

```
NETLIST
   │
   ├─[parse]──────────────────────────────────────────────────────────────────────┐
   │                                                                               │
   │  BUILD (once)                                                                 │
   │                                                                               │
   │  NumPy              Transformer 1             NumPy          Transformer 2    │
   │  ┌──────┐  A=L×U   ┌─────────────┐  A_FF⁻¹  ┌──────────┐  ┌─────────────┐  │
   │  │ A_FF │ ───────▶ │ LU Inversion│ ────────▶ │  S =     │  │  Readout    │  │
   │  │ A_FP │  baked   │ Transformer │  3n²+2n   │ -A⁻¹@A_FP│─▶│  Model      │  │
   │  └──────┘  in wts  │ (3n+2 toks) │  fwd pass │  matmul  │  │  S in wts   │  │
   │                    └─────────────┘  es        └──────────┘  └──────┬──────┘  │
   │                    ↑ discarded                                      │         │
   │                      after build                          saved to disk       │
   │                                                                      │         │
   │  INFERENCE (every query)                                             │         │
   │                                                                      ▼         │
   │                                               source voltages ──▶ Transformer 2│
   │                                                                  (2N+4 tokens) │
   │                                                                      │         │
   │                                                                  pred_V        │
   └───────────────────────────────────────────────────────────────────────────────┘
```

---

## KEY NUMBERS AT A GLANCE

| What | Value |
|---|---|
| LU token sequence length | 3·n_free + 2 per column |
| Total LU forward passes | n_free × (3·n_free + 2) |
| Readout token sequence length | 2·N + 4 |
| Voltage quantization step | 50 mV (500 units / 10000 scale) |
| LU build time (n_free=1) | ~0.2s |
| LU build time (n_free=38) | ~10s |
| LU infer time (n_free=1) | ~1.9s |
| LU infer time (n_free=38) | ~11.5s |
| Readout inference time | 10–130ms |
| LU accuracy vs numpy | ≤ 4.6e-14 (machine precision) |
| 154-circuit sweep result | 154/154 PASS |

---

## WHAT IS NUMPY DOING VS WHAT IS THE TRANSFORMER DOING

```
                    NUMPY                          TRANSFORMER
                 ─────────────                  ─────────────────
BUILD            • Build conductance            • Runs 3n+2 forward
                   matrix A from                  passes to invert A_FF
                   resistor values               • Each token = one row
                                                  of triangular sub
                 • LU factorize A_FF             • L, U values are frozen
                   (Doolittle, one-shot)           into token embeddings
                                                  — no learning, no grad
                 • S = -A_FF⁻¹ @ A_FP           • MILP assigns each
                   (plain matmul after             dimension to a
                   inversion)                      residual slot
                                                  analytically

INFERENCE        Nothing                        • Reads source voltages
                                                  from KV cache
                                                • Computes dot product
                                                  S · v_sources in FFN
                                                • Scores output vocab
                                                  to predict v_target
```
