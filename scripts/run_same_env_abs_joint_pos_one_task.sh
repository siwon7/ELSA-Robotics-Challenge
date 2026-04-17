#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task}"
GPU="${2:?gpu}"
EPOCHS="${3:-50}"
ENV_ID="${4:-0}"
RUN_NAME="${5:-bcpolicy_abs_joint_pos_e50_env0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/prepare_live_eval_env.sh"
conda activate "$ELSA_ENV_NAME"

RESULT_ROOT="$REPO_ROOT/results/abs_joint_pos_same_env_probes"
CKPT_ROOT="$REPO_ROOT/model_checkpoints/abs_joint_pos_same_env_probes"
LOG_ROOT="$REPO_ROOT/logs/abs_joint_pos_same_env_probes"
mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

LOG_PATH="$LOG_ROOT/${TASK}_${RUN_NAME}.log"

CUDA_VISIBLE_DEVICES="$GPU" python "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py" \
  --task "$TASK" \
  --env-id "$ENV_ID" \
  --epochs "$EPOCHS" \
  --eval-episodes 5 \
  --device cuda:0 \
  --run-name "$RUN_NAME" \
  --output-root "$RESULT_ROOT" \
  --checkpoint-root "$CKPT_ROOT" \
  2>&1 | tee "$LOG_PATH"
