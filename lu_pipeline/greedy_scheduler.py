"""
T1.3 — Greedy slot-assignment baseline vs MILP for LU substitution.

Simulates the slot assignment for LU forward+backward substitution
with n_free nodes using greedy strategies, compares against MILP
d_model (read from the existing results CSV).

The goal is to benchmark whether the MILP's width advantage over
greedy is significant for LU's highly regular dependency structure.

Greedy strategies implemented:
  1. first_fit    — assign each dimension to the lowest-indexed free slot
                    (slot freed when dim's last consumer has run)
  2. round_robin  — cycle through slots in rotation, evicting oldest dimension
                    if needed
"""
import csv
import os
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LU_CSV = os.path.join(BASE, "lu_pipeline", "results", "results_main_lu_direct.csv")
OUT_FIG = os.path.join(BASE, "tmlr", "figures", "fig_milp_vs_greedy.pdf")


# ── LU dependency model ─────────────────────────────────────────────────────
#
# Tokens: [start | fwd_0 .. fwd_{n-1} | bck_{n-1} .. bck_0 | halt]
# Each fwd_i produces y_i; needs b[i] and y_j for j<i
# Each bck_i produces x_i; needs y_i and x_j for j>i
# Each x_i is needed by x_j for all j < i (backward sub) and by halt
#
# A "dimension" is a named scalar that occupies a residual-stream slot
# between its birth time (token step) and death time (last consumer step).
# The "width" at any token step c is the number of live dims at that step.
#
# MILP minimizes max width.  We simulate first-fit and round-robin.

def lu_lifetimes(n):
    """
    Return a list of (birth, death, name) for all dims in LU sub with n free nodes.
    Token timeline (0-indexed steps):
      0            = start
      1..n         = fwd_0 .. fwd_{n-1}
      n+1..2n      = bck_{n-1} .. bck_0
      2n+1         = halt
    For ReGLU slots: each subtraction y_i -= L[i,j]*y_j requires a ReGLU pair;
    we model each pair as 2 transient slots that live from fwd_i to fwd_i (born
    and die in the same step, but still occupy a slot at that step).
    """
    total_steps = 2 * n + 2
    dims = []

    # b[i] : born at step 0 (start), dies at fwd_i step (i+1)
    for i in range(n):
        dims.append((0, i + 1, f"b[{i}]"))

    # y[i] : born at fwd_i step (i+1), dies at bck_i step (n+1+(n-1-i) = 2n-i)
    for i in range(n):
        bck_step = n + 1 + (n - 1 - i)  # bck processes in reverse: bck_{n-1} first
        dims.append((i + 1, bck_step, f"y[{i}]"))

    # x[i] : born at bck_i step (2n-i), dies at halt (2n+1)
    # (all x[i] are read at halt for final readout)
    for i in range(n):
        bck_step = n + 1 + (n - 1 - i)
        dims.append((bck_step, total_steps - 1, f"x[{i}]"))

    # ReGLU transient slots for forward substitution:
    # y_i needs sum over j<i of L[i,j]*y_j.  Each product is a ReGLU pair (2 slots).
    # They are born and consumed at step i+1 (forward sub step i).
    # We model them as a block of 2*i transient dims at step i+1.
    for i in range(1, n):
        for _ in range(2 * i):   # i products, 2 slots each
            dims.append((i + 1, i + 1, f"reglu_fwd_{i}_"))

    # ReGLU transient slots for backward substitution:
    # x_i needs sum over j>i of U[i,j]*x_j.  That's (n-1-i) products = 2*(n-1-i) slots.
    for i in range(n - 1):
        bck_step = n + 1 + (n - 1 - i)
        n_prods = n - 1 - i
        for _ in range(2 * n_prods):
            dims.append((bck_step, bck_step, f"reglu_bck_{i}_"))

    # Persist slots: each y[i] needs a persist slot from fwd_i to the end of fwd phase
    # so it's available for later fwd steps and for bck phase.
    # (Already modelled by y[i] birth-death above; no extra persist slots needed here
    # since the basic model already keeps y[i] alive until bck_i.)

    return dims, total_steps


def peak_width(dims, total_steps):
    """Max number of simultaneously live dims at any step (=peak width)."""
    live_at = [0] * total_steps
    for birth, death, _ in dims:
        for t in range(birth, death + 1):
            live_at[t] += 1
    return max(live_at)


def first_fit_width(dims, total_steps):
    """
    Greedy first-fit: assign dims to lowest available slot.
    A slot is freed one step after the dim's death.
    Returns the total number of slots needed (= d_model proxy).
    """
    slot_free_at = {}   # slot -> step when it becomes free again
    next_new_slot = 0
    max_slot = 0

    for birth, death, name in sorted(dims, key=lambda x: x[0]):
        # find the lowest slot free at or before birth
        chosen = None
        for s in range(next_new_slot):
            if slot_free_at.get(s, 0) <= birth:
                chosen = s
                break
        if chosen is None:
            chosen = next_new_slot
            next_new_slot += 1
        slot_free_at[chosen] = death + 1
        if chosen > max_slot:
            max_slot = chosen

    return max_slot + 1  # slots are 0-indexed


def round_robin_width(dims, total_steps):
    """
    Greedy round-robin: cycle through slots, picking next available.
    Returns the number of slots used.
    """
    slot_free_at = {}
    rr_ptr = [0]
    next_new_slot = [0]
    max_slot = [0]

    for birth, death, name in sorted(dims, key=lambda x: x[0]):
        # search from rr_ptr forward for first free slot
        start = rr_ptr[0]
        chosen = None
        for offset in range(next_new_slot[0] + 1):
            s = (start + offset) % max(1, next_new_slot[0])
            if slot_free_at.get(s, 0) <= birth:
                chosen = s
                rr_ptr[0] = (s + 1) % max(1, next_new_slot[0] + 1)
                break
        if chosen is None:
            chosen = next_new_slot[0]
            next_new_slot[0] += 1
            rr_ptr[0] = 0
        slot_free_at[chosen] = death + 1
        if chosen > max_slot[0]:
            max_slot[0] = chosen

    return max_slot[0] + 1


# ── Load MILP results ───────────────────────────────────────────────────────

rows = list(csv.DictReader(open(LU_CSV)))
milp_by_nfree = {}
for r in rows:
    if r.get("n_free") and r.get("d_model"):
        nf = int(r["n_free"])
        dm = int(r["d_model"])
        milp_by_nfree[nf] = dm  # takes last seen value per n_free (same for each)

nfree_range = sorted(milp_by_nfree.keys())

# ── Compute greedy widths ───────────────────────────────────────────────────

results = []
print(f"{'n_free':>8}  {'MILP d_model':>13}  {'Peak live':>10}  {'FirstFit':>9}  {'RoundRobin':>11}  {'FF/MILP':>8}")
print("-" * 75)
for n in nfree_range:
    dims, total_steps = lu_lifetimes(n)
    peak = peak_width(dims, total_steps)
    ff   = first_fit_width(dims, total_steps)
    rr   = round_robin_width(dims, total_steps)
    milp = milp_by_nfree[n]
    results.append((n, milp, peak, ff, rr))
    print(f"{n:>8}  {milp:>13}  {peak:>10}  {ff:>9}  {rr:>11}  {ff/milp:>8.2f}")

# ── Figure ─────────────────────────────────────────────────────────────────
ns    = [r[0] for r in results]
milps = [r[1] for r in results]
ffs   = [r[3] for r in results]
rrs   = [r[4] for r in results]

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(ns, milps, "b-",  linewidth=2,   label="MILP (optimal)")
ax.plot(ns, ffs,   "r--", linewidth=1.5, label="Greedy first-fit")
ax.plot(ns, rrs,   "g:",  linewidth=1.5, label="Greedy round-robin")
ax.set_xlabel("Free nodes ($n_{\\rm free}$)", fontsize=10)
ax.set_ylabel("$d_{\\rm model}$ (slots)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
print(f"\nSaved: {OUT_FIG}")

# print ratio summary
ratios = [r[3] / r[1] for r in results if r[0] >= 5]
print(f"\nFirstFit/MILP ratio (n_free>=5): mean={sum(ratios)/len(ratios):.2f}, "
      f"min={min(ratios):.2f}, max={max(ratios):.2f}")
