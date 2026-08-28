#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    blocks_ranking_rgb
    move_can_pot
    pick_dual_bottles
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
