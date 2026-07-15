"""
Analyse the d-sweep checkpoints produced by sweep_d.sh.

Loads:
  results/compressed_d{D}_jacobi_CKT_0001.pt  for D in {4, 8, 12, 16, 24, 32}

Produces:
  results/sweep_curves.png         : 3-panel plot of L_out, L_layer, argmax_match vs step, per d
  results/sweep_summary.png         : final L_layer and argmax_match vs d
  results/sweep_summary.csv          : tidy table

Usage:
    python analyze_sweep.py             # local: assumes results/ has the .pt files pulled
    python analyze_sweep.py --pull      # SFTP-pull from server first
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


D_VALUES = [4, 8, 12, 16, 24, 32]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def pull_from_server():
    """SFTP the .pt files from the server's results dir to local."""
    import sys
    sys.path.insert(0, r"craft")
    from _remote_setup import connect

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = connect()
    sftp = client.open_sftp()
    remote_dir = "compression/results"
    for d in D_VALUES:
        for fn in [f"compressed_d{d}_jacobi_CKT_0001.pt", f"sweep_d{d}.log"]:
            remote = f"{remote_dir}/{fn}"
            local = RESULTS_DIR / fn
            try:
                sftp.get(remote, str(local))
                print(f"pulled {fn}")
            except IOError as e:
                print(f"  miss {fn}: {e}")
    sftp.close()
    client.close()


def load_run(d: int):
    p = RESULTS_DIR / f"compressed_d{d}_jacobi_CKT_0001.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def make_plots():
    runs = {d: load_run(d) for d in D_VALUES}
    runs = {d: r for d, r in runs.items() if r is not None}
    if not runs:
        print("no runs loaded; nothing to plot")
        return

    # ---------------- per-step curves ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cmap = plt.colormaps.get_cmap("viridis")
    norm = lambda d: cmap((d - min(D_VALUES)) / (max(D_VALUES) - min(D_VALUES)))

    for d, r in runs.items():
        log = r["log"]
        steps = [e["step"] for e in log]
        l_out = [e["L_out"] for e in log]
        l_layer = [e["L_layer"] for e in log]
        match = [e["argmax_match"] for e in log]
        c = norm(d)
        axes[0].plot(steps, l_out, label=f"d={d}", color=c)
        axes[1].plot(steps, l_layer, label=f"d={d}", color=c)
        axes[2].plot(steps, match, label=f"d={d}", color=c, marker="o", ms=3, lw=0.5)

    axes[0].set_yscale("log"); axes[0].set_title("$L_{out}$ (MSE on softmax)"); axes[0].set_xlabel("step")
    axes[1].set_title("$L_{layer}$ (1 - cosine, avg)"); axes[1].set_xlabel("step")
    axes[2].set_title("argmax-match (1 = compressed pred = base pred)"); axes[2].set_xlabel("step")
    axes[2].set_ylim(-0.1, 1.1)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Compression sweep: CKT_0001, discrete Jacobi, max_seq=500, 1000 steps")
    fig.tight_layout()
    out1 = RESULTS_DIR / "sweep_curves.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    # ---------------- summary vs d ----------------
    summary = []
    for d, r in runs.items():
        log = r["log"]
        last = log[-1]
        # also report the BEST L_layer over the run, not just final
        best_l_layer = min(e["L_layer"] for e in log)
        # and the final argmax_match (most recent)
        summary.append({
            "d": d, "final_L_out": last["L_out"], "final_L_layer": last["L_layer"],
            "best_L_layer": best_l_layer, "final_argmax_match": last["argmax_match"],
        })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ds = [s["d"] for s in summary]
    axes[0].plot(ds, [s["final_L_layer"] for s in summary], "o-", label="final L_layer")
    axes[0].plot(ds, [s["best_L_layer"] for s in summary], "s--", label="best L_layer over run")
    axes[0].set_xlabel("compressed dim d (full D=36)")
    axes[0].set_ylabel("1 - cosine similarity"); axes[0].set_title("layer reconstruction loss vs d")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ds, [s["final_argmax_match"] for s in summary], "o-", c="C2")
    axes[1].set_xlabel("compressed dim d"); axes[1].set_ylabel("compressed pred = base pred")
    axes[1].set_title("argmax-match (binary)")
    axes[1].set_ylim(-0.1, 1.1); axes[1].grid(alpha=0.3)

    fig.suptitle("Compression behaviour vs d, CKT_0001 / discrete Jacobi")
    fig.tight_layout()
    out2 = RESULTS_DIR / "sweep_summary.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")

    # ---------------- CSV ----------------
    out_csv = RESULTS_DIR / "sweep_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"wrote {out_csv}")
    print()
    print("summary:")
    for s in summary:
        print(f"  d={s['d']:3d}  final_L_layer={s['final_L_layer']:.4f}  "
              f"best_L_layer={s['best_L_layer']:.4f}  match={s['final_argmax_match']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pull", action="store_true", help="SFTP-pull results from server first.")
    args = p.parse_args()
    if args.pull:
        pull_from_server()
    make_plots()
