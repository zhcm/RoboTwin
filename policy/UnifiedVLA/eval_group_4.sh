#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
PORT="$2"
SIM_GPU="$3"

TASKS=(
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    move_stapler_pad
    open_laptop
    pick_diverse_bottles
    place_bread_basket
    place_burger_fries
    place_object_basket
    put_object_cabinet
    stack_blocks_two
)

for TASK_NAME in "${TASKS[@]}"; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$PORT" "$SIM_GPU"
done
