#!/usr/bin/env bash
# Autonomous RB-SOR v2 experiment suite (4 phases) with HullKVCache.
#
# Phases (run sequentially; each is independent — failure of one does NOT
# abort the rest):
#   E1V01 — RB-SOR + V_STEP=0.01 + omega_cap=1.5 + auto-fallback to Jacobi
#   E18   — omega sweep on the 14 regression circuits at V_STEP=0.05
#   E19   — RB-SOR T-scaling on 9 rescues + 14 regressions
#   E20   — Hull vs Standard benchmark on heat 10x10 (N=101)
#
# Designed to be launched in tmux:
#   tmux new-session -d -s rbsor_v2 \
#     'bash /craft/experiments/run_rbsor_v2.sh'
#
# Survives SSH disconnect. Logs land in:
#   ~/craft_release/craft/rbsor_results/rbsor_v2_run.log
# CSV outputs in same folder.
#
# IMPORTANT: this script will NOT touch existing tmux sessions, other users,
# or any directory outside the project tree.

set -uo pipefail

ROOT=.
EXPDIR=$ROOT/craft/experiments
RBSOR_RESULTS=$ROOT/craft/rbsor_results
LOG=$RBSOR_RESULTS/rbsor_v2_run.log
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RBSOR_RESULTS" "$RBSOR_RESULTS/figures"
cd "$EXPDIR"

source $ROOT/venv/bin/activate
export PATH=$HOME/.local/bin:$PATH
export CRAFT_DATASET=$ROOT/dataset/circuit_dataset_rv.jsonl

: > "$LOG"

echo "=== RBSOR-V2 START $(date -Iseconds) ===" | tee -a "$LOG"
echo "tmux session: ${TMUX:-?}" | tee -a "$LOG"
echo "user: $(whoami)  pwd: $(pwd)" | tee -a "$LOG"
echo "venv python: $(which python)" | tee -a "$LOG"
echo "torch: $(python -c 'import torch;print(torch.__version__)')" | tee -a "$LOG"
echo "ninja: $(which ninja)" | tee -a "$LOG"

# Hull preflight.
echo "" | tee -a "$LOG"
echo "=== hull preflight ===" | tee -a "$LOG"
python -c "import _bootstrap; from transformer_vm.attention.hull_cache import HullKVCache; HullKVCache(1,1); print('hull OK')" 2>&1 | tee -a "$LOG"
HULL_RC=${PIPESTATUS[0]}
if [ "$HULL_RC" -ne 0 ]; then
    echo "FATAL hull preflight failed (rc=$HULL_RC)" | tee -a "$LOG"
    echo "FINISHED_EXIT=2 $(date -Iseconds)" | tee -a "$LOG"
    sleep 600
    exit 2
fi

# rbsor preflight.
echo "" | tee -a "$LOG"
echo "=== rbsor preflight ===" | tee -a "$LOG"
python -c "
import sys
sys.path.insert(0, '$ROOT/craft/rbsor')
sys.path.insert(0, '$ROOT/craft')
import _bootstrap
from coloring import two_color
from rbsor_reference import omega_opt, auto_T_rbsor, rbsor_solve_parsed, detect_divergence
from rbsor_interpreter import RBSORCircuitMachine
from rbsor_tokenize import tokenize_rbsor
print('rbsor OK')
" 2>&1 | tee -a "$LOG"
RBSOR_RC=${PIPESTATUS[0]}
if [ "$RBSOR_RC" -ne 0 ]; then
    echo "FATAL rbsor preflight failed (rc=$RBSOR_RC)" | tee -a "$LOG"
    echo "FINISHED_EXIT=3 $(date -Iseconds)" | tee -a "$LOG"
    sleep 600
    exit 3
fi

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

# --- The four experiments ---------------------------------------------------
# E1V01 takes ~4hrs (per user estimate) due to V_STEP=0.01 (5x bigger vocab,
# bigger sequence) plus T_floor=500 vs 200, on 154 circuits.
phase "E1V01 (rbsor_run_v01.py)"        python -u rbsor_run_v01.py

# E20 first among the small ones — single grid, gives us the hull-scaling
# headline data point as soon as possible.
phase "E20 (rbsor_heat_hull_bench.py)"  python -u rbsor_heat_hull_bench.py

# E18 — omega sweep (~14 circuits x 5 omegas = 70 builds+runs, ~30 min).
phase "E18 (rbsor_omega_sweep.py)"      python -u rbsor_omega_sweep.py

# E19 — T scaling (23 circuits x 6 T values = 138 builds+runs, ~1 hr).
phase "E19 (rbsor_tscaling.py)"         python -u rbsor_tscaling.py

echo "" | tee -a "$LOG"
echo "=== ALL PHASES DONE $(date -Iseconds) ===" | tee -a "$LOG"
echo "FINISHED_EXIT=0 $(date -Iseconds)" | tee -a "$LOG"

# Brief summary.
echo "" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
for f in results_main_rbsor_v01.csv omega_sweep_E18.csv rbsor_tscaling_E19.csv heat_hull_E20.csv; do
    if [ -f "$RBSOR_RESULTS/$f" ]; then
        echo "  $f : $(wc -l < $RBSOR_RESULTS/$f) lines" | tee -a "$LOG"
    fi
done

# Keep tmux pane alive for late attach.
sleep 7200
