# Compression infrastructure (T3)

Tracr §5-style compression of compiled CRAFT transformers, scaffolded.

## Files

- [compressed_transformer.py](compressed_transformer.py) — `CompressedTransformer` wrapper (model-agnostic). Trainable rank-`d` projection `W ∈ ℝ^(d × D)`; `W^T` on residual reads, `W` on residual writes; everything else frozen.
- [train.py](train.py) — AdamW training loop with `L_out + λ·L_layer` loss, sweep-ready.
- [README.md](README.md) — this file.

## Status

**Scaffolded but BLOCKED on user input.** See questions Q1–Q6 in paper_data.md §6 T3 (or below).

The `CompressedTransformer` wrapper is complete and model-agnostic. The training loop is complete except for two stubs that depend on which compiled model we target:
- `load_compiled_model(cfg)` — needs the per-circuit `.bin` and `.slots.json` paths
- `load_circuit_inputs(cfg)` — needs the token sequence for the circuit
- `capture_base_residuals(base, ...)` — needs hooks into base model layers

## Open questions (asked in conversation; answers needed before running)

1. **Compression subject:** LU-direct compiled (recommended) vs Jacobi vs iJacobi vs Direct-Sensitivity.
2. **Circuit identity:** start with CKT_0001? or a representative set of 3?
3. **Path layout:** local Windows or server?
4. **Compute:** GPU recommended; CPU also works.
5. **dtype:** float32 acceptable for the wrapper while base stays float64?
6. **KV cache:** disable during training (use full attention)?

Once answered: fill the three stubs in `train.py` and run `python train.py`.
