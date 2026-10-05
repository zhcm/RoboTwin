#!/usr/bin/env bash
set -euo pipefail

source /file_system/vepfs/algorithm/chenming.zhang/miniconda3/etc/profile.d/conda.sh
conda activate robotwin

CKPT_SETTING="$1"
TASK_CONFIG="$2"
PORT="$3"
SIM_GPU="$4"
shift 4
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
    TASKS=($(cut -d: -f1 task_config/_eval_step_limit.yml))
fi

for TASK_NAME in "${TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES="$SIM_GPU" PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config policy/UnifiedVLA/deploy_policy.yml \
        --overrides \
        --task_name "$TASK_NAME" \
        --task_config "$TASK_CONFIG" \
        --ckpt_setting "$CKPT_SETTING" \
        --port "$PORT"
done
