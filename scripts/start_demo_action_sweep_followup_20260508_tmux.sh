#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/demo_action_sweep_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/demo_action_sweep_20260508"
SESSION_NAME="${DEMO_ACTION_SWEEP_FOLLOWUP_SESSION:-demo_action_sweep_followup_20260508}"
TARGET_SR="${DEMO_ACTION_SWEEP_TARGET_SR:-0.9}"
POLL_SEC="${POLL_SEC:-300}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

sr_ge() {
  python - "$1" "$2" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}

task_hit() {
  local task="$1"
  local best_file="$RESULT_ROOT/$task/BEST_ACTION.txt"
  [ -s "$best_file" ] || return 1
  grep -q "^NO_ACTION_HIT" "$best_file" && return 1
  local sr
  sr="$(awk 'NR==1 {print $2}' "$best_file")"
  [ "$(sr_ge "$sr" "$TARGET_SR")" = "1" ]
}

all_tasks_hit() {
  task_hit slide_block_to_target &&
    task_hit close_box &&
    task_hit insert_onto_square_peg &&
    task_hit scoop_with_spatula
}

wait_for_session_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[$(date '+%F %T')] waiting for $session"
    sleep "$POLL_SEC"
  done
}

launch_pass() {
  local session="$1"
  local index_split="$2"
  local screen_episodes="$3"
  local confirm_trigger="$4"

  tmux has-session -t "$session" 2>/dev/null && {
    echo "pass already active: $session"
    return 0
  }

  echo "[$(date '+%F %T')] launching pass session=$session index_split=$index_split screen=$screen_episodes confirm_trigger=$confirm_trigger"
  DEMO_ACTION_SWEEP_SESSION="$session" \
    DEMO_ACTION_SWEEP_INDEX_SPLIT="$index_split" \
    DEMO_ACTION_SWEEP_SCREEN_EPISODES="$screen_episodes" \
    DEMO_ACTION_SWEEP_CONFIRM_TRIGGER_SR="$confirm_trigger" \
    bash "$SCRIPT_DIR/start_demo_action_sweep_20260508_tmux.sh"
}

followup() {
  cd "$REPO_ROOT"
  exec > >(tee -a "$LOG_ROOT/followup.log") 2>&1
  echo "=== ACTION SWEEP FOLLOWUP START target_sr=$TARGET_SR $(date '+%F %T') ==="

  wait_for_session_done demo_action_sweep_20260508
  if all_tasks_hit; then
    echo "all tasks already hit target after train pass"
    return 0
  fi

  launch_pass demo_action_sweep_full_20260508 full 5 0.8
  wait_for_session_done demo_action_sweep_full_20260508
  if all_tasks_hit; then
    echo "all tasks hit target after full-index pass"
    return 0
  fi

  launch_pass demo_action_sweep_train_long_20260508 train 10 0.7
  wait_for_session_done demo_action_sweep_train_long_20260508
  if all_tasks_hit; then
    echo "all tasks hit target after train-long pass"
    return 0
  fi

  echo "ACTION_SWEEP_FOLLOWUP_EXHAUSTED target_sr=$TARGET_SR"
  find "$RESULT_ROOT" -maxdepth 2 -name BEST_ACTION.txt -print -exec sed -n '1p' {} \;
}

launch_followup() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "followup session already exists: $SESSION_NAME"
    return 0
  }
  tmux new-session -d -s "$SESSION_NAME" -n followup \
    "cd '$REPO_ROOT' && bash '$0' --followup"
  echo "started followup: $SESSION_NAME"
  echo "logs: $LOG_ROOT/followup.log"
}

case "${1:-}" in
  --followup)
    followup
    ;;
  *)
    launch_followup
    ;;
esac
