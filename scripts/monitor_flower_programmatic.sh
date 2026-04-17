#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${1:-flower_abs_joint_pos_prog_v1}"
RUN_TAG="${2:-abs-jpos-prog-v1}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-1800}"
ELSA_ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="${ELSA_ARTIFACT_ROOT}/logs/flower_abs_joint_pos_programmatic"
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
    echo "--- nvidia-smi ---"
    nvidia-smi
    echo "--- pmon ---"
    nvidia-smi pmon -c 1 || true
    echo "--- programmatic runners ---"
    pgrep -a -f "python scripts/run_flower_programmatic_one_task.py" || true
    echo "--- client debug ---"
    find /tmp/elsa_flower_client_debug -maxdepth 1 -type f -name 'client_*.json' -print -exec cat {} \; 2>/dev/null || true
    echo "--- memory ---"
    free -h
  } >> "$MONITOR_LOG" 2>&1

  sleep "$MONITOR_INTERVAL_SEC"
done
