#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <config_path> <run_prefix> [session_name]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$1"
RUN_PREFIX="$2"
SESSION_NAME="${3:-same_env_multitask_${RUN_PREFIX}}"

ENV_ID="${ENV_ID:-0}"
EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="$ARTIFACT_ROOT/logs/same_env_multitask/${RUN_PREFIX}"
mkdir -p "$LOG_ROOT"

declare -a GPUS=("0" "1" "2" "3")
declare -a TASKS=(
  "close_box"
  "insert_onto_square_peg"
  "scoop_with_spatula"
  "slide_block_to_target"
)

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "gpu0"

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  task="${TASKS[$idx]}"
  window_name="gpu${gpu}"
  run_name="${task}_${RUN_PREFIX}_e${EPOCHS}_s${SEED}"
  log_path="$LOG_ROOT/${run_name}.log"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
  fi

  cmd="cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_config_one_task.sh' '$task' '$gpu' '$CONFIG_PATH' '$EPOCHS' '$ENV_ID' '$run_name' '$EVAL_EPISODES' '$SEED' 2>&1 | tee '$log_path'"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" C-m
done

echo "started tmux session: $SESSION_NAME"
echo "config: $CONFIG_PATH"
echo "run prefix: $RUN_PREFIX"
echo "logs: $LOG_ROOT"
