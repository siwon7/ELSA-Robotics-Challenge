#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}/logs/resume_20260506"
POLL_SEC="${POLL_SEC:-300}"

mkdir -p "$LOG_ROOT"

training_or_main_queue_running() {
  pgrep -f "scripts/train_same_env_bcpolicy_probe.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_action_ablation_queue_tmux.sh --worker" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_action_ablation_queue_tmux.sh --waiter" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_jpabs_seedsweep_queue_tmux.sh --worker" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_jpabs_seedsweep_queue_tmux.sh --waiter" >/dev/null 2>&1 && return 0
  return 1
}

queue_running() {
  local script_name="$1"
  pgrep -f "scripts/train_same_env_bcpolicy_probe.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/${script_name} --worker" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/${script_name} --waiter" >/dev/null 2>&1 && return 0
  return 1
}

wait_until_clear() {
  local label="$1"
  shift
  while "$@"; do
    echo "[$(date '+%F %T')] waiting for $label; sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

main() {
  cd "$REPO_ROOT"
  exec > >(tee -a "$LOG_ROOT/leftover_waiter.log") 2>&1

  echo "=== leftover resume waiter start $(date '+%F %T') ==="
  wait_until_clear "action_ablation/jpabs queues" training_or_main_queue_running

  echo "[$(date '+%F %T')] launching overnight leftover queue"
  bash "$REPO_ROOT/scripts/start_overnight_queue_pending_tmux.sh"
  wait_until_clear "overnight leftover queue" queue_running start_overnight_queue_pending_tmux.sh

  echo "[$(date '+%F %T')] launching recommended followup leftover queue"
  bash "$REPO_ROOT/scripts/start_recommended_followup_queue_tmux.sh"
  wait_until_clear "recommended followup leftover queue" queue_running start_recommended_followup_queue_tmux.sh

  echo "=== leftover resume waiter done $(date '+%F %T') ==="
}

main "$@"
