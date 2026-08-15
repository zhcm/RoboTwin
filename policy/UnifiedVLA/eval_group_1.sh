#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    adjust_bottle
    move_pillbottle_pad
    place_container_plate
    place_empty_cup
    place_mouse_pad
    place_phone_stand
    press_stapler
    put_bottles_dustbin
    scan_object
    shake_bottle_horizontally
    stack_blocks_three
    stamp_seal
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
