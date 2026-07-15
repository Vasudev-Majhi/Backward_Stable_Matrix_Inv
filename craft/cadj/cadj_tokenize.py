"""SPICE netlist -> CADJ token sequence.

Layout (stride 2 throughout, exactly 2N tokens per iteration block):

  pos 0                          : start
  pos 2*i+1, 2*i+2  for i in 0..N-1 : (skip, v_init(node i))
  pos 2N + 1 + 2(t*N + n)        : up_<n>_t<phase(t)>     (n in 0..N-1, t in 0..T-1)
  pos 2N + 2 + 2(t*N + n)        : <PRED>
  pos 2(T+1)*N + 1               : readout
  pos 2(T+1)*N + 2               : <PRED>
  pos 2(T+1)*N + 3               : halt

Update tokens are indexed by (node, phase). Phases group iterations:
  phase 0: t = 0          (special k=0 coefficients)
  phase 1: t = 1
  phase 2: t in [2, 3]
  phase 3: t in [4, 7]
  phase 4: t in [8, 15]
  phase 5: t in [16, 31]
  phase 6: t >= 32        (steady-state coefficients)

Within each phase, all iterations share the same (alpha, beta, gamma).
Coefficients have converged to ~1e-4 precision by t=32 so 7 phases suffice.
"""
from __future__ import annotations

import _path  # noqa: F401 — sibling sys.path shim, must be first

from parse import ParsedCircuit, parse_netlist
from cadj_reference import (
    K_LEVELS, SCALE, V_STEP, auto_T_cadj, chebyshev_coefficients,
)

PREDICTED = "<PRED>"

# Phase → (lo, hi) inclusive iteration range. hi = None means "open-ended".
PHASE_GROUPS: list[tuple[int, int | None]] = [
    (0, 0),
    (1, 1),
    (2, 3),
    (4, 7),
    (8, 15),
    (16, 31),
    (32, None),
]
NUM_PHASES = len(PHASE_GROUPS)


def phase_of(t: int) -> int:
    """Map iteration index t -> phase index in [0, NUM_PHASES)."""
    for p, (lo, hi) in enumerate(PHASE_GROUPS):
        if hi is None or lo <= t <= hi:
            return p
    return NUM_PHASES - 1   # unreachable; here for type-checker happiness


def coefficients_per_phase(
    alphas: list[float], betas: list[float], gammas: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Average each phase's per-iteration coefficients down to one (a, b, g).

    For non-degenerate spectra the Chebyshev coefficients converge to a steady
    state; averaging within a phase loses <1e-4 of accuracy in the late phases
    and is exact for the singleton phases (0, 1).
    """
    a_phase = [0.0] * NUM_PHASES
    b_phase = [0.0] * NUM_PHASES
    g_phase = [0.0] * NUM_PHASES
    counts = [0] * NUM_PHASES
    T = len(alphas)
    for t in range(T):
        p = phase_of(t)
        a_phase[p] += alphas[t]
        b_phase[p] += betas[t]
        g_phase[p] += gammas[t]
        counts[p] += 1
    for p in range(NUM_PHASES):
        if counts[p] > 0:
            a_phase[p] /= counts[p]
            b_phase[p] /= counts[p]
            g_phase[p] /= counts[p]
        else:
            # No iterations in this phase (T < phase's lo). Use the previous
            # phase's coefficients (or identity = plain Jacobi) as fallback.
            if p > 0:
                a_phase[p], b_phase[p], g_phase[p] = a_phase[p-1], b_phase[p-1], g_phase[p-1]
            else:
                a_phase[p], b_phase[p], g_phase[p] = 1.0, 0.0, 0.0
    return a_phase, b_phase, g_phase


def _v_token_for(
    voltage_scaled_int: int, v_step: int = V_STEP, k_levels: int = K_LEVELS,
) -> str:
    k = round(voltage_scaled_int / v_step)
    k = max(0, min(k_levels - 1, k))
    return f"v_{k}"


def tokenize_cadj(
    netlist: str,
    target_node: int,
    T: int | None = None,
    lam_min: float | None = None,
    lam_max: float | None = None,
    v_step: int | None = None,
    k_levels: int | None = None,
) -> tuple[list[str], ParsedCircuit, int, float, float]:
    """Returns (tokens, pc, T_used, lam_min_used, lam_max_used).

    If lam_min/lam_max are None, they're auto-derived via
    experiments.spectrum.compute_eigenvalue_bounds(pc).
    """
    vs = V_STEP if v_step is None else int(v_step)
    kl = K_LEVELS if k_levels is None else int(k_levels)

    pc = parse_netlist(netlist)

    if lam_min is None or lam_max is None:
        from experiments.spectrum import compute_eigenvalue_bounds
        lam_min, lam_max = compute_eigenvalue_bounds(pc)

    if T is None:
        T = auto_T_cadj(pc.num_nodes, lam_min, lam_max)

    N = pc.num_nodes
    toks: list[str] = ["start"]

    # Init block in natural node order 0..N-1.
    for n in range(N):
        toks.append("skip")
        v0_scaled = int(round(pc.fixed_voltage[n] * SCALE))
        toks.append(_v_token_for(v0_scaled, vs, kl))

    # Iteration blocks. Each iter t emits 2N tokens in node order 0..N-1.
    for t in range(T):
        p = phase_of(t)
        for n in range(N):
            toks.append(f"up_{n}_t{p}")
            toks.append(PREDICTED)

    # Readout + final prediction + halt.
    toks.append("readout")
    toks.append(PREDICTED)
    toks.append("halt")

    return toks, pc, T, lam_min, lam_max


if __name__ == "__main__":
    import json
    import os
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CRAFT_DATASET",
        "dataset/circuit_dataset_rv.jsonl",
    )
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            c = json.loads(line)
            toks, pc, T, lmin, lmax = tokenize_cadj(c["Netlist"], int(c["Target_Node"]))
            n_pred = sum(1 for t in toks if t == PREDICTED)
            print(f"{c['ID']} N={pc.num_nodes} T={T} lam=[{lmin:.4f},{lmax:.4f}] "
                  f"total={len(toks)} pred={n_pred}")
            print(f"  head: {toks[:14]}")
