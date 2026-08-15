#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    click_bell
    move_playingcard_away
    open_microwave
    place_a2b_right
    place_bread_skillet
    place_dual_shoes
    place_object_scale
    place_object_stand
    place_shoe
    shake_bottle
    stack_bowls_three
    turn_switch
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
