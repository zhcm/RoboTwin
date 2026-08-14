#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="$1"
TASK_CONFIG="$2"
GPU_ID="$3"

POLICY_NAME="UnifiedVLA"
CKPT_SETTING="robotwin_v1_ckpt14_joint"
SEED=0
HOST="127.0.0.1"
PORT=8001
INSTRUCTION_TYPE="unseen"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONWARNINGS=ignore::UserWarning \
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
