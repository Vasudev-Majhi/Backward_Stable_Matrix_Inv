#!/bin/bash
# d-sweep for compression on CKT_0001 (discrete Jacobi compiled).
# Run inside tmux session "compression_sweep".

set -u
cd compression
export CRAFT_DATASET=dataset/circuit_dataset_rv.jsonl
PY=venv/bin/python

CKT=CKT_0001
STEPS=1000
MAX_SEQ=500      # captures init + ~80 iters + readout; balances cost vs structure
WD=0             # wd hurt smoke; disable
LL=1.0           # let L_layer drive optimization (L_out is near-saturated and tiny)

echo "[sweep] start $(date)"
echo "[sweep] circuit=$CKT steps=$STEPS max_seq=$MAX_SEQ wd=$WD lambda_layer=$LL"

for D in 4 8 12 16 24 32; do
    LOG=results/sweep_d${D}.log
    echo "[sweep] === d=$D === $(date)"
    $PY -u server_train.py "$CKT" \
        --d-compressed "$D" --steps "$STEPS" --max-seq "$MAX_SEQ" \
        --device cuda --weight-decay "$WD" --lambda-layer "$LL" \
        > "$LOG" 2>&1
    rc=$?
    echo "[sweep] d=$D done rc=$rc $(date)"
done

echo "[sweep] all done $(date)"
