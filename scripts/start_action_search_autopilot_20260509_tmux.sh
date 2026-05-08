#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="$ARTIFACT_ROOT/logs/action_search_autopilot_20260509"
SESSION_NAME="${ACTION_SEARCH_AUTOPILOT_SESSION:-action_search_autopilot_20260509}"
POLL_SEC="${ACTION_SEARCH_AUTOPILOT_POLL_SEC:-600}"
TARGET_SR="${ACTION_SEARCH_AUTOPILOT_TARGET_SR:-0.9}"
MIN_OUTSTANDING="${ACTION_AUTOPILOT_MIN_OUTSTANDING_PER_QUEUE:-2}"
MAX_ADD="${ACTION_AUTOPILOT_MAX_ADD:-8}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

mkdir -p "$LOG_ROOT"

start_manager_if_missing() {
  local session="$1"
  local gpu="$2"
  local queue="$3"
  tmux has-session -t "$session" 2>/dev/null && return 0
  echo "[$(date '+%F %T')] starting manager session=$session gpu=$gpu queue=$queue"
  ACTION_SEARCH_MANAGER_SESSION="$session" \
    ACTION_SEARCH_MANAGER_GPU="$gpu" \
    ACTION_SEARCH_MANAGER_QUEUE="$queue" \
    ACTION_SEARCH_MANAGER_IDLE_AFTER_DONE_SEC=120 \
    ACTION_SEARCH_MANAGER_POLL_SEC=120 \
    ELSA_ENV_NAME="$ENV_NAME" \
    bash "$SCRIPT_DIR/start_action_search_manager_20260508_tmux.sh"
}

ensure_managers() {
  start_manager_if_missing \
    action_search_manager_20260508 \
    3 \
    "$SCRIPT_DIR/action_search_manager_20260508_queue.tsv"
  start_manager_if_missing \
    action_search_manager_gpu0_20260508 \
    0 \
    "$SCRIPT_DIR/action_search_manager_20260508_gpu0_queue.tsv"
  start_manager_if_missing \
    action_search_manager_gpu1_20260508 \
    1 \
    "$SCRIPT_DIR/action_search_manager_20260508_gpu1_queue.tsv"
  start_manager_if_missing \
    action_search_manager_gpu2_20260508 \
    2 \
    "$SCRIPT_DIR/action_search_manager_20260508_gpu2_queue.tsv"
}

autopilot_loop() {
  cd "$REPO_ROOT"
  exec > >(tee -a "$LOG_ROOT/autopilot.log") 2>&1
  echo "=== ACTION SEARCH AUTOPILOT START target_sr=$TARGET_SR $(date '+%F %T') ==="
  while true; do
    ensure_managers
    /home/cvlab-dgx/anaconda3/envs/"$ENV_NAME"/bin/python \
      "$SCRIPT_DIR/action_search_autopilot_20260509.py" \
      --repo-root "$REPO_ROOT" \
      --artifact-root "$ARTIFACT_ROOT" \
      --target-sr "$TARGET_SR" \
      --min-outstanding-per-queue "$MIN_OUTSTANDING" \
      --max-add "$MAX_ADD"
    echo "[$(date '+%F %T')] autopilot sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

launch() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }
  tmux new-session -d -s "$SESSION_NAME" -n autopilot \
    "cd '$REPO_ROOT' && env ACTION_SEARCH_AUTOPILOT_TARGET_SR='$TARGET_SR' ACTION_AUTOPILOT_MIN_OUTSTANDING_PER_QUEUE='$MIN_OUTSTANDING' ACTION_AUTOPILOT_MAX_ADD='$MAX_ADD' ACTION_SEARCH_AUTOPILOT_POLL_SEC='$POLL_SEC' ELSA_ENV_NAME='$ENV_NAME' bash '$0' --worker"
  echo "started action search autopilot: $SESSION_NAME"
  echo "logs: $LOG_ROOT/autopilot.log"
}

case "${1:-}" in
  --worker)
    autopilot_loop
    ;;
  *)
    launch
    ;;
esac
