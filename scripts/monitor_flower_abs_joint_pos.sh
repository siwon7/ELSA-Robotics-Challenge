#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${1:-flower_abs_joint_pos_v1}"
RUN_TAG="${2:-abs-jpos-flwr-smoke-v1}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-1800}"
LOG_ROOT="${REPO_ROOT}/logs/flower_abs_joint_pos"
MONITOR_LOG="${LOG_ROOT}/monitor_${RUN_TAG}.log"

mkdir -p "$LOG_ROOT"

while true; do
  {
    echo "===== $(date --iso-8601=seconds) ====="
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      tmux list-windows -t "${SESSION_NAME}"
    else
      echo "tmux session ${SESSION_NAME} not found"
    fi

    if [[ -f /home/cvlab-dgx/.flwr/local-superlink/state.db ]]; then
      sqlite3 /home/cvlab-dgx/.flwr/local-superlink/state.db \
        "select run_id,pending_at,running_at,finished_at,sub_status,details from run order by pending_at desc limit 12;"
    else
      echo "state db not found"
    fi
  } >> "$MONITOR_LOG"

  sleep "$MONITOR_INTERVAL_SEC"
done
