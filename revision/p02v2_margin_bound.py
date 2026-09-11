"""P0.2 v2 -- selector-exactness bound, rebuilt to answer W7.

W7 said: Lemma 1 had no proof, its constant c was undefined (so the table was not
reproducible from the lemma), the bound was loose, and -- decisively -- it gave
t_max = 17 for float16 while failure occurs at t = 1, because the lemma modelled
only PRECISION and not the format's DYNAMIC RANGE.

Corrected statement. The compiler scales the QUERY by K (weights.py: the query
projection carries HARD_K * sqrt(d_h); the key projection does not), so with

    k_p = (2p,  -p^2 + alpha*g(p)),      q_t = K' * (t, 1),     K' = K*sqrt(d_h)

the score is

    s(p) = K' * (2pt - p^2 + alpha*g(p)) = K' * (t^2 - (t-p)^2 + alpha*g(p)).

The t^2 term is common to all p and cancels in every pairwise comparison.

EXACT MARGIN. For any p != t, with 0 <= g < 1/ln2,

    s(t) - s(p) = K' * [ (t-p)^2 + alpha*(g(t) - g(p)) ]  >=  K' * (1 - alpha/ln2),

since (t-p)^2 >= 1 and |g(t)-g(p)| < 1/ln2. Write M := 1 - alpha/ln2.

ROUNDING. Each score is a 2-term dot product. Under the standard model
fl(a op b) = (a op b)(1+delta), |delta| <= u, two products and one sum give

    |fl(s(p)) - s(p)| <= gamma_2 * S_p,    gamma_2 = 2u/(1-2u),
    S_p := |2pt| + |-p^2 + alpha g(p)|   (scaled by K'),

the absolute-value sum, which is the quantity that actually bounds cancellation
error. For p, t <= P we have S_p <= K'(2P^2 + P^2 + alpha/ln2) <= 3K'P^2 for
P >= 1.

PRECISION CONDITION. fl preserves the argmax if the exact margin beats the worst
combined rounding error, i.e. if

    K' * M  >  gamma_2 * (S_t + S_p),     which is implied by     M > 6*gamma_2*P^2.

Since gamma_2 <= 2u/(1-2u) ~ 2u, a sufficient condition is

    M > 12 u P^2         =>        P < P_prec := sqrt( M / (12u) ).           (1)

K' CANCELS. The selector scale has no effect on exactness whatsoever.

RANGE CONDITION (the part v1 omitted). The scores must be representable:
max_p |s(p)| <= 3 K' P^2 must not exceed the format's largest finite value Omega,

    P < P_range := sqrt( Omega / (3 K') ).                                     (2)

Unlike (1), condition (2) DOES depend on K'. The valid position range is
min(P_prec, P_range). This is the precise sense in which K = 1e10 is a pure
liability: it cannot buy exactness (it cancels from (1)) and it can only shrink
the usable range (it divides (2)).

BOUNDED-MAGNITUDE ENCODING DOES NOT HELP -- hypothesis H2a is FALSE.

It is tempting to conclude that the bfloat16 failure is an artifact of unbounded
score magnitude and that normalizing positions (p -> p/P) would fix it. It does
not, and the reason is worth stating because it generalizes.

Normalizing rescales the margin by exactly the same factor it rescales the
magnitudes: with p_n = p/P,

    s(t) - s(p) = K' * (t-p)^2 / P^2  >=  K'/P^2,        S_p = O(K'),

so the condition becomes 1/P^2 > gamma_2, i.e. P < 1/sqrt(2u) -- still O(1/sqrt(u)),
the same scaling as (1) and within a small constant of it. Normalization buys
nothing because what must be resolved is the RELATIVE difference between adjacent
parabolic keys, which is ~1/P^2 however the family is scaled.

The honest general statement is therefore a LOWER BOUND on precision, not a
scaling bug:

    any quadratic positional key family that must distinguish adjacent positions
    up to P requires u = O(1/P^2), i.e. about 2*log2(P) significand bits,
    irrespective of the scale constant.

We verify this at the score level: the normalized encoding fails at essentially
the same position as the production encoding in every format tested.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

ALPHA = 0.3
LN2 = math.log(2.0)
M = 1.0 - ALPHA / LN2

FORMATS = {
    "float64":  dict(u=2.0 ** -53, omega=np.finfo(np.float64).max),
    "float32":  dict(u=2.0 ** -24, omega=float(np.finfo(np.float32).max)),
    "float16":  dict(u=2.0 ** -11, omega=float(np.finfo(np.float16).max)),
    "bfloat16": dict(u=2.0 ** -9,  omega=3.3895313892515355e38),
}


def g(p):
    return np.log1p(p) / LN2 % (1.0 / LN2)


def p_prec(u):
    return math.sqrt(M / (12.0 * u))


def p_range(omega, Kp):
    return math.sqrt(omega / (3.0 * Kp))


def score(p, t, Kp, normalized=False, P=None):
    p = np.asarray(p, dtype=np.float64)
    if normalized:
        pn, tn = p / P, t / P
        return Kp * (2 * pn * tn - pn ** 2 + ALPHA * g(p) / (P ** 2))
    return Kp * (2 * p * t - p ** 2 + ALPHA * g(p))


def empirical_first_failure(np_dtype, Kp, normalized=False, t_hi=4_000_000):
    """Smallest t whose argmax over a local window is not t, in the given format."""
    def ok(t):
        P = max(t, 1)
        p = np.arange(max(0, t - 3), t + 4, dtype=np.float64)
        s = score(p, t, Kp, normalized, P)
        sd = s.astype(np_dtype).astype(np.float64)
        if not np.all(np.isfinite(sd)):
            return False
        return int(p[int(np.argmax(sd))]) == t
    if ok(t_hi):
        return None
    lo, hi = 1, t_hi
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


def bf16_round(x):
    import torch
    return (torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
            .to(torch.bfloat16).to(torch.float64).numpy())


def empirical_first_failure_bf16(Kp, normalized=False, t_hi=100000):
    def ok(t):
        P = max(t, 1)
        p = np.arange(max(0, t - 3), t + 4, dtype=np.float64)
        s = bf16_round(score(p, t, Kp, normalized, P))
        if not np.all(np.isfinite(s)):
            return False
        return int(p[int(np.argmax(s))]) == t
    t = 1
    while t < t_hi and ok(t):
        t += 1
    return t if t < t_hi else None


def main():
    out = {"alpha": ALPHA, "M_exact_margin_lower_bound": M,
           "note": "K cancels from the precision condition and only shrinks the range condition"}

    # verify the exact-margin lower bound M empirically over a wide t range
    ts = np.arange(1, 300000, 1013)
    gaps = []
    for t in ts:
        cand = np.array([t - 1, t + 1], dtype=np.float64)
        gaps.append(np.min((t - cand) ** 2 + ALPHA * (g(float(t)) - g(cand))))
    out["min_exact_gap_over_t"] = float(np.min(gaps))
    out["bound_M_holds"] = bool(np.min(gaps) >= M - 1e-12)

    K_PROD = 1e10
    for Kname, Kp in (("production_K1e10", K_PROD), ("K1e4", 1e4), ("K1", 1.0)):
        tab = {}
        for fname, f in FORMATS.items():
            pp, pr = p_prec(f["u"]), p_range(f["omega"], Kp)
            tab[fname] = {
                "P_precision": pp, "P_range": pr,
                "P_max_predicted": min(pp, pr),
                "binding_constraint": "precision" if pp < pr else "range",
            }
        out[f"bound_{Kname}"] = tab

    # empirical failure positions, production encoding
    emp = {}
    for fname, dt in (("float64", np.float64), ("float32", np.float32),
                      ("float16", np.float16)):
        emp[fname] = empirical_first_failure(dt, K_PROD)
    emp["bfloat16"] = empirical_first_failure_bf16(K_PROD)
    out["empirical_first_failure_production"] = emp

    # ratio: how conservative is the (sufficient) bound where precision binds?
    cmp = {}
    for fname in FORMATS:
        pred = out["bound_production_K1e10"][fname]["P_max_predicted"]
        e = emp.get(fname)
        cmp[fname] = {
            "predicted_P_max": pred, "empirical_first_failure": e,
            "conservatism_factor": (e / pred) if (e and pred) else None,
            "binding": out["bound_production_K1e10"][fname]["binding_constraint"],
        }
    out["bound_vs_empirical"] = cmp

    # H2a: bounded-magnitude (normalized) encoding
    norm = {}
    for fname, dt in (("float32", np.float32), ("float16", np.float16)):
        norm[fname] = empirical_first_failure(dt, K_PROD, normalized=True)
    norm["bfloat16"] = empirical_first_failure_bf16(K_PROD, normalized=True)
    out["normalized_encoding_first_failure"] = norm
    # H2a verdict: normalization rescales margin and magnitude equally, so the
    # predicted ceiling stays O(1/sqrt(u)) -- it does NOT become P-independent.
    out["normalized_encoding_predicted_P"] = {
        f: 1.0 / math.sqrt(2 * FORMATS[f]["u"]) for f in FORMATS
    }
    out["H2a_verdict"] = {
        "hypothesis": "bounded-magnitude encoding restores correct selection in bf16",
        "supported": False,
        "reason": ("normalizing positions divides the margin by P^2 exactly as it "
                   "divides the magnitudes, leaving the relative gap unchanged; "
                   "the required precision is u = O(1/P^2) for any quadratic key "
                   "family, independent of the scale constant"),
        "evidence": "normalized_encoding_first_failure vs empirical_first_failure_production",
    }
    # bits of significand needed to reach position P
    out["design_rule_bits_for_P"] = {
        str(P): 2 * math.log2(P) + math.log2(12 / M) for P in (10, 100, 601, 10000)
    }

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "p02v2_margin_bound.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
