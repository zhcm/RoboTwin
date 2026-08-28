#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    put_bottles_dustbin
    beat_block_hammer
    lift_pot
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
