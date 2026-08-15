#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    blocks_ranking_size
    click_alarmclock
    hanging_mug
    lift_pot
    move_can_pot
    pick_dual_bottles
    place_a2b_left
    place_can_basket
    place_cans_plasticbox
    place_fan
    rotate_qrcode
    stack_bowls_two
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
