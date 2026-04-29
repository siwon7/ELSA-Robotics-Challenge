#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <session_name> <run_suffix> <gpu_csv> <task1:config1> [<task2:config2> ...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION_NAME="$1"
RUN_SUFFIX="$2"
GPU_CSV="${3// /}"
shift 3

EPOCHS="${EPOCHS:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"
ENV_ID="${ENV_ID:-0}"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="$ARTIFACT_ROOT/logs/long_pair_sweep/${RUN_SUFFIX}"
mkdir -p "$LOG_ROOT"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"

if [ "${#GPUS[@]}" -ne "$#" ]; then
  echo "GPU count (${#GPUS[@]}) must match task:config pair count ($#)" >&2
  exit 1
fi

declare -a TASKS=()
declare -a CONFIG_PATHS=()

for pair in "$@"; do
  if [[ "$pair" != *:* ]]; then
    echo "invalid task:config pair: $pair" >&2
    exit 1
  fi

  task="${pair%%:*}"
  config_path="${pair#*:}"

  if [ -z "$task" ] || [ -z "$config_path" ]; then
    echo "invalid task:config pair: $pair" >&2
    exit 1
  fi

  if [[ "$config_path" != /* ]]; then
    config_path="$REPO_ROOT/$config_path"
  fi

  if [ ! -f "$config_path" ]; then
    echo "config file not found: $config_path" >&2
    exit 1
  fi

  TASKS+=("$task")
  CONFIG_PATHS+=("$config_path")
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  task="${TASKS[$idx]}"
  config_path="${CONFIG_PATHS[$idx]}"
  window_name="gpu${gpu}"
  task_basename="${task//\//_}"
  run_name="${task_basename}_${RUN_SUFFIX}_e${EPOCHS}_s${SEED}"
  log_path="$LOG_ROOT/gpu${gpu}_${task}.log"

  if [ "$idx" -eq 0 ]; then
    tmux new-session -d -s "$SESSION_NAME" -n "$window_name"
  else
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
  fi

  cmd="cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_config_one_task.sh' '$task' '$gpu' '$config_path' '$EPOCHS' '$ENV_ID' '$run_name' '$EVAL_EPISODES' '$SEED' 2>&1 | tee '$log_path'"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" C-m
done

echo "session_name=$SESSION_NAME log_dir=$LOG_ROOT"
