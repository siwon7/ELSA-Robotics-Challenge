#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <task> <config_path> <run_prefix> [session_name]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="$1"
CONFIG_PATH="$2"
RUN_PREFIX="$3"
SESSION_NAME="${4:-same_env_envsweep_${TASK}_${RUN_PREFIX}}"

EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"
ENV_IDS="${ENV_IDS:-0 1 2 3}"
GPUS="${GPUS:-0 1 2 3}"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="$ARTIFACT_ROOT/logs/same_env_envsweep/${TASK}_${RUN_PREFIX}"
mkdir -p "$LOG_ROOT"

read -r -a ENV_ID_ARRAY <<< "$ENV_IDS"
read -r -a GPU_ARRAY <<< "$GPUS"

if [ "${#ENV_ID_ARRAY[@]}" -ne "${#GPU_ARRAY[@]}" ]; then
  echo "ENV_IDS count must match GPUS count" >&2
  exit 1
fi

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "gpu${GPU_ARRAY[0]}"

for idx in "${!ENV_ID_ARRAY[@]}"; do
  env_id="${ENV_ID_ARRAY[$idx]}"
  gpu="${GPU_ARRAY[$idx]}"
  window_name="gpu${gpu}"
  run_name="${TASK}_${RUN_PREFIX}_env${env_id}_e${EPOCHS}_s${SEED}"
  log_path="$LOG_ROOT/${run_name}.log"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
  fi

  cmd="cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_config_one_task.sh' '$TASK' '$gpu' '$CONFIG_PATH' '$EPOCHS' '$env_id' '$run_name' '$EVAL_EPISODES' '$SEED' 2>&1 | tee '$log_path'"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" C-m
done

echo "started tmux session: $SESSION_NAME"
echo "task: $TASK"
echo "config: $CONFIG_PATH"
echo "run prefix: $RUN_PREFIX"
echo "env ids: $ENV_IDS"
echo "gpus: $GPUS"
echo "logs: $LOG_ROOT"
