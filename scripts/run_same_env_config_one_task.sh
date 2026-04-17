#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task}"
GPU="${2:?gpu}"
CONFIG_PATH="${3:?config_path}"
EPOCHS="${4:-50}"
ENV_ID="${5:-0}"
RUN_NAME="${6:?run_name}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/prepare_live_eval_env.sh"
conda activate "${ELSA_ENV_NAME:-elsa_challenge}"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/same_env_suite"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/same_env_suite"
LOG_ROOT="$ARTIFACT_ROOT/logs/same_env_suite"
mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

LOG_PATH="$LOG_ROOT/${RUN_NAME}.log"

CUDA_VISIBLE_DEVICES="$GPU" python "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py" \
  --task "$TASK" \
  --dataset-config-path "$CONFIG_PATH" \
  --env-id "$ENV_ID" \
  --epochs "$EPOCHS" \
  --eval-episodes 5 \
  --device cuda:0 \
  --run-name "$RUN_NAME" \
  --output-root "$RESULT_ROOT" \
  --checkpoint-root "$CKPT_ROOT" \
  2>&1 | tee "$LOG_PATH"
