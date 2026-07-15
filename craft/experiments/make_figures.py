"""Generate paper figures from the CSVs produced by run_experiments.py.

Figures:
  - phase_diagram.png (E6): scatter rho(M) vs kappa(A), coloured pass/fail
  - convergence.png (E3): abs_error vs T, log-log
  - timing_vs_S2.png (E10): transformer time vs seq_length^2
  - size_scaling.png (E8): model_size and seq_length vs N
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def _load_csv(name: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x, default=None):
    try:
        v = float(x)
        # reject inf / nan if default is numeric
        if default is not None and (v != v or v == float("inf")):
            return default
        return v
    except (ValueError, TypeError):
        return default


def phase_diagram():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("results_main.csv")
    if not rows:
        log.warning("no results_main.csv; skipping phase diagram")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Separate pass / fail, handle inf
    for passfail, colour, marker in (("PASS", "#2ca02c", "o"), ("FAIL", "#d62728", "x")):
        xs, ys = [], []
        for r in rows:
            if r.get("pass_fail") != passfail:
                continue
            rho = _to_float(r.get("rho_M"))
            kappa = _to_float(r.get("kappa_A"))
            if rho is None or kappa is None:
                continue
            if rho == float("inf") or kappa == float("inf"):
                continue
            # Avoid log(1 - rho) = log(0) when rho == 1
            one_minus_rho = max(1e-12, 1.0 - rho)
            xs.append(one_minus_rho)
            ys.append(kappa)
        ax.scatter(xs, ys, c=colour, marker=marker, s=30, alpha=0.7, label=f"{passfail} ({len(xs)})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1 - \rho(M)$  (distance from Jacobi divergence)")
    ax.set_ylabel(r"$\kappa(A)$  (condition number)")
    ax.set_title("Phase diagram: Jacobi-transformer pass/fail (E6)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "phase_diagram.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def convergence():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("convergence_E3.csv")
    if not rows:
        log.warning("no convergence_E3.csv; skipping")
        return

    by_circuit: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        cid = r["circuit_id"]
        T = _to_float(r.get("T"))
        err = _to_float(r.get("abs_error"))
        if T is None or err is None:
            continue
        by_circuit.setdefault(cid, []).append((T, err))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid, data in sorted(by_circuit.items()):
        data.sort()
        Ts = [t for t, _ in data]
        errs = [max(1e-8, e) for _, e in data]
        ax.plot(Ts, errs, marker="o", label=cid)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations T")
    ax.set_ylabel("Absolute error (V)")
    ax.set_title("Error vs T for 5 circuits (E3)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="tolerance")
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "convergence.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def timing_vs_s2():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("results_main.csv")
    if not rows:
        log.warning("no results_main.csv; skipping timing plot")
        return

    xs_dsl, ys_dsl, xs_tf, ys_tf = [], [], [], []
    for r in rows:
        S = _to_float(r.get("seq_length"))
        dsl_t = _to_float(r.get("dsl_time_ms"))
        tf_t = _to_float(r.get("transformer_time_ms"))
        if S and dsl_t:
            xs_dsl.append(S * S)
            ys_dsl.append(dsl_t)
        if S and tf_t:
            xs_tf.append(S * S)
            ys_tf.append(tf_t)

    fig, ax = plt.subplots(figsize=(8, 6))
    if xs_dsl:
        ax.scatter(xs_dsl, ys_dsl, c="#1f77b4", marker="o", s=20, alpha=0.6,
                   label=f"DSL evaluator ({len(xs_dsl)})")
    if xs_tf:
        ax.scatter(xs_tf, ys_tf, c="#ff7f0e", marker="x", s=40, alpha=0.8,
                   label=f"Compiled transformer ({len(xs_tf)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sequence length squared, $S^2$")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Inference time vs $S^2$ (E10)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "timing_vs_S2.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def size_scaling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("results_main.csv")
    if not rows:
        return

    Ns, sizes, vocabs, seqs = [], [], [], []
    for r in rows:
        n = _to_float(r.get("N"))
        sz = _to_float(r.get("model_size_bytes"))
        vo = _to_float(r.get("vocab"))
        sq = _to_float(r.get("seq_length"))
        if n is None:
            continue
        Ns.append(n)
        sizes.append(sz)
        vocabs.append(vo)
        seqs.append(sq)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.scatter(Ns, [s for s in sizes if s], alpha=0.6)
    ax.set_xlabel("N (# nodes)")
    ax.set_ylabel("Model size (bytes)")
    ax.set_title("E8: Model size vs N")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(Ns, [v for v in vocabs if v], alpha=0.6, c="#ff7f0e")
    ax.set_xlabel("N (# nodes)")
    ax.set_ylabel("Vocab size")
    ax.set_title("E8: Vocab size vs N")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(Ns, [s for s in seqs if s], alpha=0.6, c="#2ca02c")
    ax.set_xlabel("N (# nodes)")
    ax.set_ylabel("Sequence length S")
    ax.set_title("E9: Seq length vs N (T per tier)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "size_scaling.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def vstep_ablation():
    """E12: pass count vs V_STEP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("vstep_ablation_E12.csv")
    if not rows:
        log.warning("no vstep_ablation_E12.csv; skipping")
        return

    # Group by V_STEP
    from collections import defaultdict
    by_vstep: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        v = _to_float(r.get("V_STEP_V"))
        if v is None:
            continue
        by_vstep[v].append(r)

    v_steps = sorted(by_vstep)
    pass_counts = [sum(1 for r in by_vstep[v] if str(r.get("pass", "")).lower() == "true")
                   for v in v_steps]
    total = [len(by_vstep[v]) for v in v_steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [f"{v}V" for v in v_steps]
    ax.bar(xs, pass_counts, color="#2ca02c", alpha=0.75)
    for x, p, t in zip(xs, pass_counts, total):
        ax.text(x, p + 0.2, f"{p}/{t}", ha="center", fontsize=10)
    ax.set_xlabel("V_STEP (quantization step)")
    ax.set_ylabel("Circuits passing")
    ax.set_title("E12: Mode B pass rate vs quantization step")
    ax.set_ylim(0, max(total) + 1.5 if total else 11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "vstep_ablation.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def tscaling():
    """E13: abs_error vs T, overlaid with theoretical ρ^T curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("tscaling_E13.csv")
    if not rows:
        log.warning("no tscaling_E13.csv; skipping")
        return

    # Group by circuit
    from collections import defaultdict
    by_cid: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        T = _to_float(r.get("T"))
        err = _to_float(r.get("abs_error"))
        rho = _to_float(r.get("rho_M"))
        if T is None or err is None or rho is None:
            continue
        by_cid[r["circuit_id"]].append((T, err, rho))

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    cids = sorted(by_cid)
    for i, cid in enumerate(cids):
        data = sorted(by_cid[cid])
        Ts = [t for t, _, _ in data]
        errs = [max(1e-8, e) for _, e, _ in data]
        rho = data[0][2]
        color = cmap(i / max(1, len(cids) - 1))

        # Empirical
        ax.plot(Ts, errs, marker="o", linewidth=1.8, color=color,
                label=f"{cid} (ρ={rho:.3f})")

        # Theoretical curve: err_theo(T) = err(T=1) * rho^(T-1), only if rho<1
        if rho > 0 and rho < 1 and errs[0] > 0:
            err_theo = [max(1e-8, errs[0] * (rho ** (t - 1))) for t in Ts]
            ax.plot(Ts, err_theo, "--", linewidth=1.0, color=color, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations T")
    ax.set_ylabel("Absolute error (V)")
    ax.set_title("E13: Error vs T — empirical (solid) vs theoretical ρ^T (dashed)")
    ax.axhline(0.05, color="gray", linestyle=":", alpha=0.5, label="tolerance 0.05V")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "tscaling.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def failure_modes():
    """E14: bar chart of failure-mode breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("failure_modes_E14.csv")
    if not rows:
        log.warning("no failure_modes_E14.csv; skipping")
        return

    from collections import Counter
    modes = Counter(r["mode"] for r in rows)
    # Keep PASS at top, then failure modes ordered by count
    fail_modes = {m: c for m, c in modes.items() if m != "PASS"}
    pass_count = modes.get("PASS", 0)

    order = ["PASS", "A_non_convergent", "B_compound", "B_simple", "C_boundary", "UNPARSEABLE"]
    labels = [m for m in order if m in modes]
    counts = [modes[m] for m in labels]
    colors = {"PASS": "#2ca02c", "A_non_convergent": "#d62728",
              "B_compound": "#ff7f0e", "B_simple": "#ffbb78",
              "C_boundary": "#9467bd", "UNPARSEABLE": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts, color=[colors.get(m, "#888") for m in labels])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1, str(c),
                ha="center", fontsize=10)
    ax.set_ylabel("# circuits")
    ax.set_title(f"E14: Failure-mode breakdown  ({pass_count} PASS / "
                 f"{sum(fail_modes.values())} FAIL)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "failure_modes.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def mode_a_tscaling():
    """E17: error vs T for Mode A circuit — DSL AND Jacobi ref both fail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("mode_a_tscaling_E17.csv")
    if not rows:
        log.warning("no mode_a_tscaling_E17.csv; skipping")
        return

    from collections import defaultdict
    by_cid = defaultdict(list)
    for r in rows:
        T = _to_float(r.get("T"))
        dsl = _to_float(r.get("err_dsl_vs_truth"))
        jac = _to_float(r.get("err_jacobi_vs_truth"))
        if T is None or dsl is None or jac is None:
            continue
        by_cid[r["circuit_id"]].append((T, dsl, jac, r.get("rho_M", "")))

    fig, ax = plt.subplots(figsize=(9, 5))
    for cid, data in sorted(by_cid.items()):
        data.sort()
        Ts = [t for t, _, _, _ in data]
        dsl_errs = [max(1e-8, e) for _, e, _, _ in data]
        jac_errs = [max(1e-8, e) for _, _, e, _ in data]
        rho = data[0][3]
        ax.plot(Ts, dsl_errs, marker="o", linewidth=2,
                label=f"{cid} DSL (ρ={rho})")
        ax.plot(Ts, jac_errs, marker="x", linewidth=2, linestyle="--",
                label=f"{cid} Jacobi ref (ρ={rho})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations T")
    ax.set_ylabel("|error vs truth|  (V)")
    ax.set_title("E17: Mode A — DSL and Jacobi both diverge from truth at every T")
    ax.axhline(0.05, color="gray", linestyle=":", alpha=0.5, label="pass tol 0.05V")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "mode_a_tscaling.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def bcompound_vstep():
    """E15: B-compound circuits vs V_STEP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("bcompound_vstep_E15.csv")
    if not rows:
        log.warning("no bcompound_vstep_E15.csv; skipping")
        return

    from collections import defaultdict
    by_cid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        v = _to_float(r.get("V_STEP_V"))
        err = _to_float(r.get("err_vs_truth"))
        if v is None or err is None:
            continue
        by_cid[r["circuit_id"]].append((v, err))

    fig, ax = plt.subplots(figsize=(9, 5))
    for cid, data in sorted(by_cid.items()):
        data.sort(reverse=True)  # biggest V_STEP first for x-axis
        vs = [v for v, _ in data]
        errs = [max(1e-6, e) for _, e in data]
        ax.plot(vs, errs, marker="o", linewidth=2, label=cid)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # coarse → fine, left to right
    ax.set_xlabel("V_STEP (V)")
    ax.set_ylabel("|error vs truth|  (V)")
    ax.set_title("E15: B-compound error vs V_STEP")
    ax.axhline(0.05, color="gray", linestyle=":", alpha=0.5, label="pass tol 0.05V")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "bcompound_vstep.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def e16_phase_diagram():
    """E16: phase diagram using V_STEP=0.01 results (if available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv("results_main_E16.csv")
    if not rows:
        log.warning("no results_main_E16.csv; skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for passfail, colour, marker in (("PASS", "#2ca02c", "o"), ("FAIL", "#d62728", "x")):
        xs, ys = [], []
        for r in rows:
            if r.get("pass_fail") != passfail:
                continue
            rho = _to_float(r.get("rho_M"))
            kappa = _to_float(r.get("kappa_A"))
            if rho is None or kappa is None or rho == float("inf"):
                continue
            xs.append(max(1e-12, 1.0 - rho))
            ys.append(kappa)
        ax.scatter(xs, ys, c=colour, marker=marker, s=30, alpha=0.7,
                   label=f"{passfail} ({len(xs)})")

    v_step = rows[0].get("V_STEP_V", "?") if rows else "?"
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1 - \rho(M)$")
    ax.set_ylabel(r"$\kappa(A)$")
    ax.set_title(f"E16: Phase diagram at V_STEP={v_step}V (finer quantization)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "phase_diagram_E16.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info(f"wrote {out}")


def make_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    phase_diagram()
    convergence()
    timing_vs_s2()
    size_scaling()
    vstep_ablation()
    tscaling()
    failure_modes()
    bcompound_vstep()
    mode_a_tscaling()
    e16_phase_diagram()


if __name__ == "__main__":
    make_all()
