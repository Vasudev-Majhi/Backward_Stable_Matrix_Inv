# benchmarks/ — PDE + IEEE DC Power Flow Extensions

This folder extends the SPICE-Jacobi compiled-transformer to two additional
domains, **without modifying any existing file** under `craft/`.

## Contents

```
benchmarks/
├── pde/                  Heat + Poisson 2D netlist generators (V+R only, no DSL change needed)
├── ieee/                 IEEE 14/30/57-bus DC power flow → netlist with I-source elements
├── ext/                  DSL extension for current sources (wraps parse / interpreter / etc.)
├── orchestrators/        Per-domain run scripts (PDE+Jacobi, PDE+RB-SOR, IEEE+Jacobi, IEEE+RB-SOR)
├── runner_ext.py         Single entry point that swaps in extended modules per algorithm
└── results/              Output CSVs and per-domain dataset JSONLs
```

## What's new vs. the existing 154-SPICE pipeline

### 1. PDE (zero DSL change)

`pde/heat_2d.py` reuses the existing `experiments/heat_netlist.py`'s
`thermal_grid_to_spice()` for heat-equation grids. `pde/poisson_2d.py` adds a
Poisson-equation generator that uses I-sources for the forcing term — so it
needs the I-source extension.

### 2. IEEE DC power flow (needs I-source extension)

DC power flow:  `B · θ = P`. We map:

- branches (i,j) with reactance `x_ij`  →  resistor `R = x_ij` between nodes i,j
- non-slack bus i with net injection P_i  →  current source `I = P_i` from i to GND
- slack bus  →  fixed at 0V (= GND, node 0)

`ieee/ieee_to_netlist.py` produces the netlist; `ieee/ieee_cases.py` embeds
the MATPOWER bus/branch/gen arrays for case14, case30, case57.

### 3. The I-source DSL extension (`ext/`)

A current source `I<name> N+ 0 <amperes>` with `N+ = n` adds a constant
`I_inj_norm[n] = I_inj_raw[n] / Σ_g_raw[n]` (in volts) to the Jacobi update:

```
V_new[n] = (Σ w_k · V_nbr[k] + I_inj_norm[n] · SCALE) / SCALE
         = Σ w_k · V_nbr[k] / SCALE   +  I_inj_norm[n]
```

Implementation (subclasses, no edits to existing files):

- `ext/parse_isource.py`        — wraps `parse.parse_netlist`, scans for `I` lines.
- `ext/tokenize_isource.py`     — wraps tokenize; fills `i_inj_slot` per update token.
- `ext/interpreter_isource.py`  — adds one `InputDimension("i_inj_slot")` and one
                                  extra term to the Jacobi `v_new_free` persist.
- `ext/interpreter_rbsor_isource.py` — same addition but inside the RB-SOR
                                       `v_jacobi` step (before SOR relaxation).
- `ext/build_isource.py`        — builds models using the extended classes.

The existing `interpreter.CircuitMachine` and `rbsor_interpreter.RBSORCircuitMachine`
are untouched; the extension subclasses them and overrides `build()`.

## Running

### Local smoke test

```bash
# from craft
python benchmarks/orchestrators/run_pde_jacobi.py     --smoke
python benchmarks/orchestrators/run_pde_rbsor.py      --smoke
python benchmarks/orchestrators/run_ieee_jacobi.py    --smoke
python benchmarks/orchestrators/run_ieee_rbsor.py     --smoke
```

### Full sweep on the Linux server (recommended)

```bash
# from craft (Windows side)
python benchmarks/server/push_and_launch.py push
python benchmarks/server/push_and_launch.py launch
python benchmarks/server/push_and_launch.py status
python benchmarks/server/push_and_launch.py pull   # after DONE
```

uses for the existing pipeline.

## Verification expected

1. **Regression check:** running the existing `craft/run_experiments.py`
   (unchanged) reproduces the existing 154-circuit results bit-identically.
2. **I-source smoke:** a 4-node hand-crafted circuit with one I-source
   matches numpy direct solve to within `V_STEP = 0.05V`.
3. **PDE 5×5 sanity:** symmetric BC solution matches the existing E11 result.
4. **IEEE 14-bus:** angle predictions within 0.01 rad of the published MATPOWER
   reference for case14.
5. **PDE 10×10 + RB-SOR + Hull KV + V_STEP=0.01:** previously-failing heat
   diffusion case passes (err < 0.05V).
