"""154-circuit sweep for idea3: end-to-end LU inference pipeline (server version)."""
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
RESULTS_DIR  = os.path.join(_CLAUDE_FILES, "cadj_results")

FIELDS = [
    "circuit_id", "complexity", "N", "n_free", "n_fixed", "target_node",
    "pred_V", "truth_V", "abs_error", "pass_fail",
    "build_time_s", "infer_time_s", "build_status", "error",
]


def _write_rows(rows: list[dict], out_csv: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse
    from build_idea3 import build_for_circuit
    from run_idea3 import run_circuit_idea3

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    circuits: list[dict] = []
    with open(DATASET_PATH) as f:
        for line in f:
            circuits.append(json.loads(line))
    if args.limit is not None:
        circuits = circuits[: args.limit]

    out_csv = os.path.join(RESULTS_DIR, f"results_idea3{args.suffix}.csv")
    print(f"[IDEA3] {len(circuits)} circuits  out={out_csv}", flush=True)

    rows: list[dict] = []
    pass_n = fail_n = err_n = 0

    for i, c in enumerate(circuits):
        cid = c["ID"]
        row = {fn: "" for fn in FIELDS}
        row["circuit_id"]  = cid
        row["complexity"]  = c.get("Complexity", "")
        row["truth_V"]     = c["Ground_Truth_Vout"]
        row["target_node"] = c["Target_Node"]

        try:
            tb0 = time.time()
            sidecar_json = os.path.join(_HERE, "sidecars", cid, "sidecar.json")
            needs_build  = args.force_rebuild or not os.path.exists(sidecar_json)
            if needs_build:
                info = build_for_circuit(cid, force=args.force_rebuild)
                row["build_status"] = "built"
            else:
                row["build_status"] = "cached"
                with open(sidecar_json) as f:
                    info = json.load(f)
            row["N"]       = info.get("N", "")
            row["n_free"]  = info.get("n_free", "")
            row["n_fixed"] = info.get("n_fixed", "")
            build_t = time.time() - tb0

            status, pred_v, truth, infer_t = run_circuit_idea3(cid, tol=args.tol)

            row["pred_V"]       = f"{pred_v:.6f}"
            row["abs_error"]    = f"{abs(pred_v - truth):.6f}"
            row["pass_fail"]    = "PASS" if status == "PASS" else "FAIL"
            row["build_time_s"] = f"{build_t:.2f}"
            row["infer_time_s"] = f"{infer_t:.2f}"

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

        print(
            f"[{i+1:>3}/{len(circuits)}] {cid:<10} "
            f"pred={row['pred_V']:>12} truth={row['truth_V']:>8.4f} "
            f"err={row['abs_error']:>10} [{row['pass_fail']}] "
            f"build={row['build_time_s']:>6}s infer={row['infer_time_s']:>6}s",
            flush=True,
        )

    print(
        f"\n[IDEA3] DONE  pass={pass_n}  fail={fail_n}  err={err_n}  "
        f"total={len(circuits)}  out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
