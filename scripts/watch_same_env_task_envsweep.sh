#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <session_name> <task> <config_path> <run_prefix>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION_NAME="$1"
TASK="$2"
CONFIG_PATH="$3"
RUN_PREFIX="$4"

EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"
ENV_IDS="${ENV_IDS:-0 1 2 3}"
GPUS="${GPUS:-0 1 2 3}"
POLL_SECONDS="${POLL_SECONDS:-120}"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"

read -r -a ENV_ID_ARRAY <<< "$ENV_IDS"
read -r -a GPU_ARRAY <<< "$GPUS"

if [ "${#ENV_ID_ARRAY[@]}" -ne "${#GPU_ARRAY[@]}" ]; then
  echo "ENV_IDS count must match GPUS count" >&2
  exit 1
fi

log() {
  printf '[watchdog] %s\n' "$*"
}

result_path_for_env() {
  local env_id="$1"
  local run_name="${TASK}_${RUN_PREFIX}_env${env_id}_e${EPOCHS}_s${SEED}"
  printf '%s/results/same_env_suite/%s/%s/env_%03d/result.json' \
    "$ARTIFACT_ROOT" "$TASK" "$run_name" "$env_id"
}

is_running_env() {
  local env_id="$1"
  local run_name="${TASK}_${RUN_PREFIX}_env${env_id}_e${EPOCHS}_s${SEED}"
  pgrep -f "train_same_env_bcpolicy_probe.py --task ${TASK} .* --env-id ${env_id} .* --run-name ${run_name}" >/dev/null
}

restart_env() {
  local env_id="$1"
  local gpu="$2"
  local window_name="gpu${gpu}"
  local run_name="${TASK}_${RUN_PREFIX}_env${env_id}_e${EPOCHS}_s${SEED}"
  local cmd="cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_config_one_task.sh' '$TASK' '$gpu' '$CONFIG_PATH' '$EPOCHS' '$env_id' '$run_name' '$EVAL_EPISODES' '$SEED'"

  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    log "session ${SESSION_NAME} missing, recreating launcher session"
    EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" ENV_IDS="$ENV_IDS" GPUS="$GPUS" \
      bash "$SCRIPT_DIR/start_same_env_task_envsweep_tmux.sh" "$TASK" "$CONFIG_PATH" "$RUN_PREFIX" "$SESSION_NAME"
    return
  fi

  if ! tmux list-windows -t "$SESSION_NAME" | grep -q "${window_name}"; then
    log "window ${window_name} missing, recreating launcher session"
    EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" ENV_IDS="$ENV_IDS" GPUS="$GPUS" \
      bash "$SCRIPT_DIR/start_same_env_task_envsweep_tmux.sh" "$TASK" "$CONFIG_PATH" "$RUN_PREFIX" "$SESSION_NAME"
    return
  fi

  log "restarting env=${env_id} gpu=${gpu} run=${run_name}"
  tmux send-keys -t "${SESSION_NAME}:${window_name}" C-c
  sleep 1
  tmux send-keys -t "${SESSION_NAME}:${window_name}" "$cmd" C-m
}

log "watching session=${SESSION_NAME} task=${TASK} run_prefix=${RUN_PREFIX} env_ids=${ENV_IDS} gpus=${GPUS}"

while true; do
  done_count=0

  for idx in "${!ENV_ID_ARRAY[@]}"; do
    env_id="${ENV_ID_ARRAY[$idx]}"
    gpu="${GPU_ARRAY[$idx]}"
    result_path="$(result_path_for_env "$env_id")"

    if [ -f "$result_path" ]; then
      done_count=$((done_count + 1))
      continue
    fi

    if is_running_env "$env_id"; then
      continue
    fi

    restart_env "$env_id" "$gpu"
  done

  if [ "$done_count" -eq "${#ENV_ID_ARRAY[@]}" ]; then
    log "all result files exist, exiting"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
