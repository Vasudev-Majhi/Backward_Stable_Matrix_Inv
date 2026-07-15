#!/usr/bin/env bash
# Autonomous CADJ experiment suite (2 phases) with HullKVCache.
#
# Phases:
#   C1     — CADJ full 154 at V_STEP=0.05  (apples-to-apples vs RB-SOR v0)
#   C1V01  — CADJ full 154 at V_STEP=0.01  (apples-to-apples vs RB-SOR v01)
#
# Survives SSH disconnect (tmux). Each phase is independent — failure of one
# does not abort the other. All outputs land in:
#   ~/craft_release/craft/cadj_results/
# Log:
#   ~/craft_release/craft/cadj_results/cadj_full_run.log
#
# Launch via:
#   tmux new-session -d -s cadj_full \
#     'bash craft/cadj/run_cadj_full.sh'

set -uo pipefail

ROOT=.
CADJ_DIR=$ROOT/craft/cadj
CLAUDE_FILES=$ROOT/craft
RESULTS=$ROOT/craft/cadj_results
LOG=$RESULTS/cadj_full_run.log

mkdir -p "$RESULTS"
# cd to craft/ so `import _bootstrap` resolves; cadj_run_all.py adjusts
# its own sys.path to include craft/cadj/ for sibling imports.
cd "$CLAUDE_FILES"

source $ROOT/venv/bin/activate
export PATH=$HOME/.local/bin:$PATH
export CRAFT_DATASET=$ROOT/dataset/circuit_dataset_rv.jsonl

: > "$LOG"

echo "=== CADJ START $(date -Iseconds) ===" | tee -a "$LOG"
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

# CADJ preflight.
echo "" | tee -a "$LOG"
echo "=== cadj preflight ===" | tee -a "$LOG"
python -c "
import sys
sys.path.insert(0, '$CADJ_DIR')
sys.path.insert(0, '$ROOT/craft')
import _bootstrap
from cadj_reference import chebyshev_coefficients, auto_T_cadj, cadj_solve_parsed
from cadj_interpreter import CADJCircuitMachine
from cadj_tokenize import tokenize_cadj
from experiments.spectrum import compute_eigenvalue_bounds
print('cadj OK')
" 2>&1 | tee -a "$LOG"
CADJ_RC=${PIPESTATUS[0]}
if [ "$CADJ_RC" -ne 0 ]; then
    echo "FATAL cadj preflight failed (rc=$CADJ_RC)" | tee -a "$LOG"
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

# C1: full sweep at V_STEP=0.05 (default). Direct head-to-head with RB-SOR v0.
phase "C1 (V_STEP=0.05)"     python -u cadj/cadj_run_all.py

# C1V01: full sweep at V_STEP=0.01 (5x finer). Direct head-to-head with RB-SOR v01.
phase "C1V01 (V_STEP=0.01)"  python -u cadj/cadj_run_all.py --v-step 100 --k-levels 2400 --suffix _v01

echo "" | tee -a "$LOG"
echo "=== ALL PHASES DONE $(date -Iseconds) ===" | tee -a "$LOG"
echo "FINISHED_EXIT=0 $(date -Iseconds)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
for f in results_main_cadj.csv results_main_cadj_v01.csv; do
    if [ -f "$RESULTS/$f" ]; then
        n=$(wc -l < "$RESULTS/$f")
        p=$(awk -F, 'NR>1 && $14=="PASS"{c++} END{print c+0}' "$RESULTS/$f")
        echo "  $f : $n lines, $p PASS" | tee -a "$LOG"
    fi
done

# Keep tmux pane alive for late attach.
sleep 7200
