#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="$1"
GPU_ID="$2"

while IFS=: read -r TASK_NAME _; do
    bash policy/UnifiedVLA/eval.sh "$TASK_NAME" "$TASK_CONFIG" "$GPU_ID"
done < task_config/_eval_step_limit.yml
