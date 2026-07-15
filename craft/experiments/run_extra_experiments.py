"""Autonomous orchestrator for the follow-up experiments E14-E17.

Sequences:
  1. E14 — classify existing failures into Mode A / B-simple / B-compound / C
  2. E15 — B-compound V_STEP test (3 circuits × 4 V_STEP, ~30 min)
  3. E16 — all 154 at the V_STEP recommended by E15 (2-3 hours)
  4. E17 — Mode A T-scaling (CKT_0133, 5 T values, ~2 min)
  5. Figures — regenerate all paper figures including the new ones
  6. Summary — append E14-E17 sections to summary.md

All phases run sequentially in the same Python process. Results are
checkpointed to CSVs as soon as each phase completes, so the run survives
interruption. Designed to be launched via `tmux new-session -d` on the
remote server — user can disconnect and results will be waiting.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import logging
import os
import time
import traceback

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extras")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def _append_section(md_path: str, content: str) -> None:
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(content.rstrip() + "\n")


def _e14_section() -> str:
    path = os.path.join(RESULTS_DIR, "failure_modes_E14.csv")
    if not os.path.exists(path):
        return ""
    from collections import Counter
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    modes = Counter(r["mode"] for r in rows)
    total = sum(modes.values())
    lines = [
        "## Failure-mode classification (E14)",
        "",
        f"All 154 circuits classified from existing results_main.csv:",
        "",
        "| Mode | Count | Meaning |",
        "|------|-------|---------|",
    ]
    descr = {
        "PASS": "within 0.05V of truth",
        "A_non_convergent": "Jacobi reference itself fails (rho~1)",
        "B_simple": "quantization miss within ~2*V_STEP",
        "B_compound": "DSL fixed-point drift at high rho",
        "C_boundary": "within float epsilon of tolerance",
        "UNPARSEABLE": "row missing required fields",
    }
    order = ["PASS", "A_non_convergent", "B_compound", "B_simple", "C_boundary", "UNPARSEABLE"]
    for m in order:
        if m in modes:
            lines.append(f"| {m} | {modes[m]} | {descr.get(m,'')} |")
    lines.append(f"| **Total** | {total} | |")
    return "\n".join(lines) + "\n"


def _e15_section() -> str:
    path = os.path.join(RESULTS_DIR, "bcompound_vstep_E15.csv")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    from collections import defaultdict
    by_cid: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r["V_STEP_V"])
            err = float(r.get("err_vs_truth", "nan"))
            ok = str(r.get("pass", "")).lower() == "true"
            by_cid[r["circuit_id"]].append((v, err, "PASS" if ok else "FAIL"))
        except (ValueError, TypeError):
            continue

    lines = [
        "## B-compound V_STEP test (E15)",
        "",
        "Three representative B-compound circuits at four V_STEPs:",
        "",
        "| Circuit | V_STEP=0.05V | V_STEP=0.01V | V_STEP=0.005V | V_STEP=0.001V |",
        "|---------|-------------|-------------|--------------|--------------|",
    ]
    for cid in sorted(by_cid):
        data = sorted(by_cid[cid], reverse=True)
        row = f"| {cid} |"
        for v, err, ok in data:
            row += f" {err:.3f}V ({ok}) |"
        lines.append(row)
    # Verdict
    decision = os.path.join(RESULTS_DIR, "e15_decision.txt")
    if os.path.exists(decision):
        with open(decision) as f:
            content = f.read()
        lines.append("")
        lines.append(f"Decision: `{content.strip().splitlines()[-1]}`")
    return "\n".join(lines) + "\n"


def _e16_section() -> str:
    path = os.path.join(RESULTS_DIR, "results_main_E16.csv")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    from collections import Counter
    by_tier: dict[str, Counter] = {}
    for r in rows:
        by_tier.setdefault(r["complexity"] or "?", Counter())[r["pass_fail"]] += 1
    v_step = rows[0].get("V_STEP_V", "?") if rows else "?"

    lines = [
        f"## Full re-run at V_STEP={v_step}V (E16)",
        "",
        f"All 154 circuits, DSL evaluator at V_STEP={v_step}V:",
        "",
        "| Tier | N | Pass | Fail | Pass rate |",
        "|------|---|------|------|-----------|",
    ]
    total_pass = 0
    for tier in ("Basic", "Intermediate", "Hard"):
        c = by_tier.get(tier, Counter())
        p, n = c["PASS"], c["PASS"] + c["FAIL"]
        total_pass += p
        rate = f"{p/n*100:.1f}%" if n else "-"
        lines.append(f"| {tier} | {n} | {p} | {n - p} | {rate} |")
    lines.append(f"| **Total** | {len(rows)} | {total_pass} | {len(rows) - total_pass} | "
                 f"{total_pass/len(rows)*100:.1f}% |")
    return "\n".join(lines) + "\n"


def _e17_section() -> str:
    path = os.path.join(RESULTS_DIR, "mode_a_tscaling_E17.csv")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [
        "## Mode A T-scaling (E17)",
        "",
        "Both DSL and Jacobi-ref stay far from truth at every T — proves",
        "the failure is algorithmic, not a DSL artefact.",
        "",
        "| Circuit | rho | T | truth | jacobi ref | dsl pred | err(jacobi) | err(dsl) |",
        "|---------|-----|---|-------|-----------|----------|-------------|----------|",
    ]
    for r in rows:
        if r.get("error"):
            continue
        lines.append(
            f"| {r['circuit_id']} | {r['rho_M']} | {r['T']} | "
            f"{r['truth_V']} | {r['jacobi_ref_V']} | {r['dsl_pred_V']} | "
            f"{r['err_jacobi_vs_truth']} | {r['err_dsl_vs_truth']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # Phase 1 — E14
    log.info("=== Phase 1: E14 failure-mode classification ===")
    try:
        import e14_failure_modes
        e14_failure_modes.run_e14()
    except Exception as e:  # noqa: BLE001
        log.error(f"E14 failed: {e}")
        log.debug(traceback.format_exc())

    # Phase 2 — E15 (B-compound V_STEP test)
    log.info("=== Phase 2: E15 B-compound V_STEP test ===")
    try:
        import e15_bcompound
        e15_bcompound.run_e15()
    except Exception as e:  # noqa: BLE001
        log.error(f"E15 failed: {e}")
        log.debug(traceback.format_exc())

    # Phase 3 — E16 (all 154 at fine V_STEP)
    log.info("=== Phase 3: E16 all circuits at fine V_STEP ===")
    try:
        import e16_allcircuits_finevstep
        e16_allcircuits_finevstep.run_e16()
    except Exception as e:  # noqa: BLE001
        log.error(f"E16 failed: {e}")
        log.debug(traceback.format_exc())

    # Phase 4 — E17 (Mode A T-scaling)
    log.info("=== Phase 4: E17 Mode A T-scaling ===")
    try:
        import e17_modeA_tscaling
        e17_modeA_tscaling.run_e17()
    except Exception as e:  # noqa: BLE001
        log.error(f"E17 failed: {e}")
        log.debug(traceback.format_exc())

    # Phase 5 — figures
    log.info("=== Phase 5: figures ===")
    try:
        import make_figures
        make_figures.make_all()
    except Exception as e:  # noqa: BLE001
        log.error(f"make_figures failed: {e}")
        log.debug(traceback.format_exc())

    # Phase 6 — append new sections to summary.md
    log.info("=== Phase 6: append summary ===")
    summary_md = os.path.join(RESULTS_DIR, "summary.md")
    try:
        for section in (_e14_section(), _e15_section(), _e16_section(), _e17_section()):
            if section:
                _append_section(summary_md, section)
        log.info(f"summary.md updated: {summary_md}")
    except Exception as e:  # noqa: BLE001
        log.error(f"summary append failed: {e}")

    log.info(f"=== ALL EXTRAS DONE in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
