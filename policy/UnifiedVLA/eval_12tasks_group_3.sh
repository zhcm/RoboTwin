#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    stack_bowls_two
    place_empty_cup
    place_a2b_left
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
