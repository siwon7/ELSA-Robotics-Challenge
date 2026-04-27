#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task}"
GPU="${2:?gpu}"
CONFIG_PATH="${3:?config_path}"
EPOCHS="${4:-50}"
TRAIN_ENV_IDS="${5:?train_env_ids_csv}"
EVAL_ENV_IDS="${6:?eval_env_ids_csv}"
RUN_NAME="${7:?run_name}"
EVAL_EPISODES="${8:-20}"
SEED="${9:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/prepare_live_eval_env.sh"
conda activate "${ELSA_ENV_NAME:-elsa_challenge}"

PYTHON_BIN="$CONDA_BASE/envs/${ELSA_ENV_NAME:-elsa_challenge}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/same_env_multi_env_suite"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/same_env_multi_env_suite"
LOG_ROOT="$ARTIFACT_ROOT/logs/same_env_multi_env_suite"
mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

LOG_PATH="$LOG_ROOT/${RUN_NAME}.log"

IFS=',' read -r -a TRAIN_ENV_IDS_ARR <<< "$TRAIN_ENV_IDS"
IFS=',' read -r -a EVAL_ENV_IDS_ARR <<< "$EVAL_ENV_IDS"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py" \
  --task "$TASK" \
  --dataset-config-path "$CONFIG_PATH" \
  --env-id "${TRAIN_ENV_IDS_ARR[0]}" \
  --train-env-ids "${TRAIN_ENV_IDS_ARR[@]}" \
  --eval-env-ids "${EVAL_ENV_IDS_ARR[@]}" \
  --epochs "$EPOCHS" \
  --eval-episodes "$EVAL_EPISODES" \
  --device cuda:0 \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --output-root "$RESULT_ROOT" \
  --checkpoint-root "$CKPT_ROOT" \
  2>&1 | tee "$LOG_PATH"
