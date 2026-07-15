"""
T1.5 — Wilson 95% CI for all binomial pass-rates in the paper.
Prints LaTeX-ready strings for inserting into main.tex.
"""
import math

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def fmt(k, n):
    lo, hi = wilson_ci(k, n)
    pct = 100 * k / n
    return f"{k}/{n} = {pct:.1f}\\% [{100*lo:.1f}\\%,\\,{100*hi:.1f}\\%]"

def fmt_plain(k, n, label):
    lo, hi = wilson_ci(k, n)
    pct = 100 * k / n
    print(f"  {label:45s}  {k}/{n} = {pct:5.1f}%  [{100*lo:5.1f}%, {100*hi:5.1f}%]")

print("=== Wilson 95% CIs for all paper binomial rates ===\n")
print(f"  {'Description':45s}  {'k/n':>10}  [lo%, hi%]")
print("-" * 75)

# main pass rates (updated to current CSV numbers)
fmt_plain(150, 154, "LU-direct PASS@50mV (paper)")
fmt_plain(151, 154, "LU-direct PASS@50mV (current CSV)")
fmt_plain(154, 154, "LU-direct PASS@75mV")
fmt_plain(150, 154, "iJacobi PASS (current CSV, updated from 135)")
fmt_plain(135, 154, "iJacobi PASS (paper, old run)")
fmt_plain(123, 154, "RBSOR PASS@50mV")
fmt_plain(96,  154, "Discrete Jacobi PASS@50mV (current CSV)")
fmt_plain(90,  154, "Discrete Jacobi PASS@50mV (paper, old run)")

# Theorem 1 rates
print()
fmt_plain(62,  90,  "Stall-radius FP rate on PASS circuits (69% ~ 62/90)")
fmt_plain(24,  24,  "Stall-radius covers ALL 24 spurious-FP failures")
fmt_plain(12,  24,  "Minkowski condition fires on failures")
fmt_plain(9,   90,  "Minkowski fires on PASS circuits (false pos)")

# cross-domain
print()
fmt_plain(601, 601, "Cross-domain PASS@50mV (601/601)")

# Trained baseline
print()
fmt_plain(7,   142, "Trained baseline PASS@50mV (7/142)")
fmt_plain(10,  142, "Trained baseline PASS@75mV (10/142)")

print("\n=== LaTeX inserts (copy into main.tex) ===\n")

rates = [
    ("LU-direct PASS@50mV",           150, 154),
    ("LU-direct PASS@75mV",           154, 154),
    ("iJacobi PASS@50mV (updated)",   150, 154),
    ("RBSOR PASS@50mV",               123, 154),
    ("Discrete Jacobi PASS@50mV",      96, 154),
    ("Stall-radius FP on PASS circs",  62,  90),
    ("Minkowski fires on failures",    12,  24),
    ("Minkowski fires on PASS",         9,  90),
    ("Cross-domain PASS",             601, 601),
    ("Trained-TF PASS@50mV",           7, 142),
]

for label, k, n in rates:
    print(f"% {label}")
    print(f"  {fmt(k, n)}")
    print()
