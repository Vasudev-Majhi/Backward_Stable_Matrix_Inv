#!/usr/bin/env bash
# Autonomous RB-SOR experiment runner with HullKVCache enabled.
#
# Designed to be launched in tmux:
#   tmux new-session -d -s rbsor_full \
#     'bash /craft/experiments/run_rbsor_full.sh'
#
# Then user can disconnect; results land in
#   ~/craft_release/craft/experiments/results/results_main_rbsor.csv
# Log: ~/craft_release/rbsor_full_run.log
#
# Survives SSH disconnect (tmux). Phases are independent — a single failure
# does not abort the rest.

set -uo pipefail

ROOT=.
EXPDIR=$ROOT/craft/experiments
RESULTS=$EXPDIR/results                 # legacy Jacobi suite output dir
RBSOR_RESULTS=$ROOT/craft/rbsor_results   # RB-SOR-specific outputs
LOG=$RBSOR_RESULTS/rbsor_full_run.log
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RBSOR_RESULTS"

cd "$EXPDIR"

# Activate venv + ensure ninja (for hull JIT) is on PATH + dataset env var.
source $ROOT/venv/bin/activate
export PATH=$HOME/.local/bin:$PATH
export CRAFT_DATASET=$ROOT/dataset/circuit_dataset_rv.jsonl

# Truncate previous log of THIS wrapper.
: > "$LOG"

echo "=== RBSOR START $(date -Iseconds) ===" | tee -a "$LOG"
echo "tmux session: ${TMUX:-?}" | tee -a "$LOG"
echo "user: $(whoami)  pwd: $(pwd)" | tee -a "$LOG"
echo "venv python: $(which python)" | tee -a "$LOG"
echo "torch: $(python -c 'import torch;print(torch.__version__)')" | tee -a "$LOG"
echo "ninja: $(which ninja)" | tee -a "$LOG"

# Sanity-check hull is loadable BEFORE we start the long loop.
echo "" | tee -a "$LOG"
echo "=== hull preflight ===" | tee -a "$LOG"
python -c "import _bootstrap; from transformer_vm.attention.hull_cache import HullKVCache; HullKVCache(1,1); print('hull OK')" 2>&1 | tee -a "$LOG"
HULL_RC=${PIPESTATUS[0]}
if [ "$HULL_RC" -ne 0 ]; then
    echo "FATAL hull preflight failed (rc=$HULL_RC); aborting" | tee -a "$LOG"
    echo "FINISHED_EXIT=2 $(date -Iseconds)" | tee -a "$LOG"
    sleep 600
    exit 2
fi

# Sanity-check rbsor modules are importable.
echo "" | tee -a "$LOG"
echo "=== rbsor preflight ===" | tee -a "$LOG"
python -c "
import sys, os
sys.path.insert(0, '$ROOT/craft/rbsor')
sys.path.insert(0, '$ROOT/craft')
import _bootstrap
from coloring import two_color
from rbsor_reference import omega_opt, auto_T_rbsor, rbsor_solve_parsed
from rbsor_interpreter import RBSORCircuitMachine
from rbsor_tokenize import tokenize_rbsor
print('rbsor OK')
" 2>&1 | tee -a "$LOG"
RBSOR_RC=${PIPESTATUS[0]}
if [ "$RBSOR_RC" -ne 0 ]; then
    echo "FATAL rbsor preflight failed (rc=$RBSOR_RC); aborting" | tee -a "$LOG"
    echo "FINISHED_EXIT=3 $(date -Iseconds)" | tee -a "$LOG"
    sleep 600
    exit 3
fi

# phase NAME CMD...
phase() {
    local name="$1"; shift
    echo "" | tee -a "$LOG"
    echo "===== PHASE [$name] start $(date -Iseconds) =====" | tee -a "$LOG"
    local t0=$(date +%s)
    "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    local dt=$(( $(date +%s) - t0 ))
    echo "===== PHASE [$name] end   $(date -Iseconds)  rc=$rc  ${dt}s =====" | tee -a "$LOG"
}

# --- The actual experiments -------------------------------------------------
# R1: full 154-circuit RB-SOR sweep with hull cache. Builds + runs each.
phase "R1 (rbsor_run_all.py — 154 circuits, hull)"  python -u rbsor_run_all.py

# Bonus: rerun Jacobi sweep with hull cache for an apples-to-apples baseline.
# The Jacobi orchestrator writes to $RESULTS/results_main.csv; we copy a
# snapshot into the RB-SOR results folder for direct comparison.
if [ -f "$RESULTS/results_main.csv" ]; then
    cp -a "$RESULTS/results_main.csv" "$RBSOR_RESULTS/results_main_jacobi_prev_${TS}.csv"
fi
phase "E1-baseline (run_experiments.py — Jacobi 154, hull, --skip-transformer-sample)" \
    python -u run_experiments.py --skip-transformer-sample
if [ -f "$RESULTS/results_main.csv" ]; then
    cp -a "$RESULTS/results_main.csv" "$RBSOR_RESULTS/results_main_jacobi_baseline.csv"
fi

echo "" | tee -a "$LOG"
echo "=== ALL PHASES DONE $(date -Iseconds) ===" | tee -a "$LOG"
echo "FINISHED_EXIT=0 $(date -Iseconds)" | tee -a "$LOG"

# Brief comparison summary.
echo "" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
if [ -f "$RBSOR_RESULTS/results_main_rbsor.csv" ]; then
    echo "RB-SOR results:" | tee -a "$LOG"
    awk -F, 'NR>1 && $17=="PASS"{p++} NR>1 && $17=="FAIL"{f++} END{print "  rbsor pass="p" fail="f}' \
        "$RBSOR_RESULTS/results_main_rbsor.csv" | tee -a "$LOG"
fi
if [ -f "$RBSOR_RESULTS/results_main_jacobi_baseline.csv" ]; then
    echo "Jacobi (E1) results:" | tee -a "$LOG"
    awk -F, 'NR>1 && $10=="PASS"{p++} NR>1 && $10=="FAIL"{f++} END{print "  jacobi pass="p" fail="f}' \
        "$RBSOR_RESULTS/results_main_jacobi_baseline.csv" | tee -a "$LOG"
fi

# Keep tmux pane alive so reconnects can see the final status.
sleep 3600
