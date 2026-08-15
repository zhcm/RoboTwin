#!/usr/bin/env bash
set -euo pipefail

source /file_system/vepfs/algorithm/chenming.zhang/miniconda3/etc/profile.d/conda.sh
conda activate robotwin

TASK_NAME="$1"
TASK_CONFIG="$2"
PORT="$3"
SIM_GPU="$4"

POLICY_NAME="UnifiedVLA"
CKPT_SETTING="robotwin_v1_ckpt14_joint"
SEED=0
HOST="127.0.0.1"
INSTRUCTION_TYPE="unseen"

CUDA_VISIBLE_DEVICES="$SIM_GPU" PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config "policy/$POLICY_NAME/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --ckpt_setting "$CKPT_SETTING" \
    --seed "$SEED" \
    --policy_name "$POLICY_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --instruction_type "$INSTRUCTION_TYPE"
