#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common_env.sh"

SESSION="${1:-elsa_watchdog_live_eval}"
LOG_DIR="$ELSA_ROOT/logs/live_eval"
LOG_FILE="$LOG_DIR/watchdog.session.log"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd \"$ELSA_ROOT\" && conda run -n \"$ELSA_ENV_NAME\" python scripts/watchdog_fk_eval.py 2>&1 | tee -a \"$LOG_FILE\""

echo "started tmux session: $SESSION"
echo "log: $LOG_FILE"
