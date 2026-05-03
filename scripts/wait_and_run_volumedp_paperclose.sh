#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/paperclose_suite"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/paperclose_suite"
LOG_ROOT="$ARTIFACT_ROOT/logs/paperclose_suite"
RUN_NAME="${RUN_NAME:-slide_volumedp_paperclose_v1_e50}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_paperclose_v1.yaml}"
QUEUE_SESSION="${QUEUE_SESSION:-overnight_pending_16}"
POLL_SEC="${POLL_SEC:-300}"

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

worker_is_running() {
  local gpu="$1"
  pgrep -f "bash .*start_overnight_queue_pending_tmux.sh --worker $gpu" >/dev/null 2>&1
}

pick_available_gpu() {
  local gpu
  for gpu in 0 1 2 3; do
    if ! worker_is_running "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

result_exists() {
  compgen -G "$RESULT_ROOT/slide_block_to_target/$RUN_NAME/*/result.json" >/dev/null
}

main() {
  local launcher_log="$LOG_ROOT/${RUN_NAME}_launcher.log"
  exec > >(tee -a "$launcher_log") 2>&1

  echo "=== $(date '+%F %T') paper-close launcher start ==="
  echo "queue_session=$QUEUE_SESSION"
  echo "config_path=$CONFIG_PATH"
  echo "run_name=$RUN_NAME"

  if result_exists; then
    echo "result already exists, nothing to do"
    exit 0
  fi

  local gpu
  until gpu="$(pick_available_gpu)"; do
    echo "[$(date '+%F %T')] all queue workers still running; sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done

  echo "[$(date '+%F %T')] picked GPU $gpu"

  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "${ELSA_ENV_NAME:-elsa_challenge}"

  local python_bin="/home/cvlab-dgx/anaconda3/envs/${ELSA_ENV_NAME:-elsa_challenge}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py" \
    --task slide_block_to_target \
    --dataset-config-path "$CONFIG_PATH" \
    --epochs 50 \
    --eval-episodes 20 \
    --device cuda:0 \
    --seed 0 \
    --run-name "$RUN_NAME" \
    --output-root "$RESULT_ROOT" \
    --checkpoint-root "$CKPT_ROOT"

  local status=$?
  echo "=== $(date '+%F %T') paper-close launcher end exit=$status ==="
  exit "$status"
}

main "$@"
