#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/recovered_live_eval_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/insert_volumedp_recovery_20260508"
SESSION_NAME="${INSERT_VOLUMEDP_RECOVERY_SESSION:-insert_volumedp_recovery_20260508}"
EPISODES="${RECOVERED_EVAL_EPISODES:-20}"
POLL_SEC="${POLL_SEC:-300}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

mkdir -p "$LOG_ROOT"

RUN_NAME="insert_volumedp_w4_eeaux_e50"
TASK="insert_onto_square_peg"
CKPT="$ARTIFACT_ROOT/model_checkpoints/overnight_queue/$TASK/$RUN_NAME/env_000.pth"
CFG="$ARTIFACT_ROOT/results/overnight_queue/$TASK/$RUN_NAME/env_000/resolved_config.yaml"

current_training_running() {
  pgrep -f "train_same_env_bcpolicy_probe.py .*--task $TASK .*--run-name $RUN_NAME" >/dev/null 2>&1
}

run_eval() {
  local mode="$1"
  local out="$RESULT_ROOT/$TASK/$RUN_NAME/$mode/result.json"
  if [ -f "$out" ]; then
    echo "skip existing eval: $out"
    return 0
  fi

  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"

  local python_bin="/home/cvlab-dgx/anaconda3/envs/${ENV_NAME}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/eval_flower_checkpoint_live.py"
    --model-path "$CKPT"
    --task "$TASK"
    --dataset-config-path "$CFG"
    --split training
    --env-ids 0
    --episodes "$EPISODES"
    --device cuda:0
    --output "$out"
  )
  if [ "$mode" = "hysteresis" ]; then
    cmd+=(
      --gripper-eval-mode hysteresis
      --gripper-open-threshold 0.65
      --gripper-close-threshold 0.35
      --gripper-min-hold-steps 2
    )
  fi

  echo "=== INSERT VOLUMEDP EVAL START mode=$mode episodes=$EPISODES $(date '+%F %T') ==="
  export CUDA_VISIBLE_DEVICES="${INSERT_VOLUMEDP_RECOVERY_GPU:-2}"
  "${cmd[@]}"
  echo "=== INSERT VOLUMEDP EVAL END mode=$mode $(date '+%F %T') ==="
}

main() {
  cd "$REPO_ROOT"
  exec > >(tee -a "$LOG_ROOT/recovery.log") 2>&1
  echo "=== INSERT VOLUMEDP RECOVERY WAITER START $(date '+%F %T') ==="
  while current_training_running; do
    echo "[$(date '+%F %T')] current $RUN_NAME training still running; sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
  while [ ! -f "$CKPT" ] || [ ! -f "$CFG" ]; do
    echo "[$(date '+%F %T')] waiting for checkpoint/config"
    echo "ckpt=$CKPT"
    echo "cfg=$CFG"
    sleep "$POLL_SEC"
  done
  run_eval threshold
  run_eval hysteresis
  echo "=== INSERT VOLUMEDP RECOVERY WAITER DONE $(date '+%F %T') ==="
}

if [ "${1:-}" != "--run" ]; then
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    exit 0
  }

  tmux new-session -d -s "$SESSION_NAME" -n recovery \
    "cd '$REPO_ROOT' && bash '$0' --run"
  echo "started session: $SESSION_NAME"
  echo "logs: $LOG_ROOT/recovery.log"
  exit 0
fi

main
