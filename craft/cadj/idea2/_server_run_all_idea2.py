"""154-circuit sweep: build + run all circuits with the idea2 LU-transformer pipeline."""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import traceback

_HERE         = os.path.dirname(os.path.abspath(__file__))
_CADJ         = os.path.dirname(_HERE)
_CLAUDE_FILES = os.path.dirname(_CADJ)
_REPO_ROOT   = os.path.dirname(_CLAUDE_FILES)

for _p in (_CLAUDE_FILES, _CADJ, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "circuit_dataset_rv.jsonl")
DATASET_PATH = os.environ.get("CRAFT_DATASET", _DEFAULT_DATASET)
MODEL_DIR    = _CLAUDE_FILES
RESULTS_DIR  = os.path.join(_CLAUDE_FILES, "cadj_results")

FIELDS = [
    "circuit_id", "complexity", "N", "n_free", "n_fixed", "target_node",
    "pred_V_dsl", "truth_V", "abs_error", "pass_fail",
    "lu_build_s", "lu_infer_s", "readout_build_s", "infer_s", "n_tokens",
    "build_status", "error",
]


def _write_rows(rows: list[dict], out_csv: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse
    from _server_build_idea2 import build_for_circuit
    from _server_run_idea2 import run_circuit_idea2

    ap = argparse.ArgumentParser(description="Idea2: LU-transformer 154-circuit sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--v-step", type=int, default=None)
    ap.add_argument("--k-levels", type=int, default=None)
    args = ap.parse_args()

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]

    out_csv = os.path.join(RESULTS_DIR, f"results_idea2{args.suffix}.csv")
    print(f"[IDEA2] {len(circuits)} circuits  out={out_csv}", flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0

    for i, c in enumerate(circuits):
        cid = c["ID"]
        row = {fn: "" for fn in FIELDS}
        row["circuit_id"] = cid
        row["complexity"]  = c.get("Complexity", "")
        row["truth_V"]     = c["Ground_Truth_Vout"]
        row["target_node"] = c["Target_Node"]
        t0 = time.time()

        try:
            model_path = os.path.join(MODEL_DIR, f"model_{cid}_idea2.bin")
            needs_build = not os.path.exists(model_path) or args.force_rebuild

            if needs_build:
                info = build_for_circuit(
                    cid, force=args.force_rebuild,
                    v_step=args.v_step, k_levels=args.k_levels,
                )
                row["build_status"]    = "built"
                row["N"]               = info.get("N", "")
                row["n_free"]          = info.get("n_free", "")
                row["n_fixed"]         = info.get("n_fixed", "")
                row["lu_build_s"]      = f"{info.get('lu_build_s', 0.0):.2f}"
                row["lu_infer_s"]      = f"{info.get('lu_infer_s', 0.0):.2f}"
                row["readout_build_s"] = f"{info.get('readout_build_s', 0.0):.2f}"
            else:
                row["build_status"] = "cached"

            status, pred_v, truth, infer_s, n_tokens = run_circuit_idea2(
                cid, tol=args.tol,
                v_step=args.v_step, k_levels=args.k_levels,
            )
            row["pred_V_dsl"] = f"{pred_v:.4f}"
            row["abs_error"]  = f"{abs(pred_v - truth):.4f}"
            row["pass_fail"]  = "PASS" if status == "PASS" else "FAIL"
            row["infer_s"]    = f"{infer_s:.2f}"
            row["n_tokens"]   = n_tokens

            if status == "PASS":
                pass_n += 1
            else:
                fail_n += 1

        except Exception as e:
            row["error"]        = repr(e)
            row["pass_fail"]    = "ERROR"
            row["build_status"] = "error"
            err_n += 1
            log.error("%s: %s\n%s", cid, e, traceback.format_exc())

        rows.append(row)
        _write_rows(rows, out_csv)

        dt = time.time() - t0
        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} "
            f"pred={row['pred_V_dsl']:>8} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>7} [{row['pass_fail']}] "
            f"lu_build={row.get('lu_build_s','--'):>6}s lu_infer={row.get('lu_infer_s','--'):>6}s "
            f"readout={row.get('readout_build_s','--'):>6}s infer={row.get('infer_s','--'):>6}s "
            f"toks={row.get('n_tokens','--')} total={dt:.1f}s",
            flush=True,
        )

    print(
        f"\n[IDEA2] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"total={len(circuits)}  out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
