#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-action_sweep_dinov3}"
SESSION_NAME="${2:-same_env_suite}"
EPOCHS="${3:-50}"
ENV_ID="${4:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_ROOT/experiments/slide_block_to_target_sameenv_suite.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to parse $MANIFEST" >&2
  exit 1
fi

TASK="$(jq -r '.task' "$MANIFEST")"
COUNT="$(jq -r --arg group "$GROUP" '.groups[$group] | length' "$MANIFEST")"
if [ "$COUNT" = "null" ] || [ "$COUNT" -le 0 ]; then
  echo "Unknown or empty group: $GROUP" >&2
  exit 1
fi

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

for IDX in $(seq 0 $((COUNT - 1))); do
  NAME="$(jq -r --arg group "$GROUP" --argjson idx "$IDX" '.groups[$group][$idx].name' "$MANIFEST")"
  CONFIG_REL="$(jq -r --arg group "$GROUP" --argjson idx "$IDX" '.groups[$group][$idx].config' "$MANIFEST")"
  CONFIG_PATH="$REPO_ROOT/$CONFIG_REL"
  WINDOW_NAME="${NAME#slide_}"
  GPU="$IDX"

  if [ "$IDX" -eq 0 ]; then
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME"
  else
    tmux new-window -t "$SESSION_NAME" -n "$WINDOW_NAME"
  fi

  tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" \
    "cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_config_one_task.sh' '$TASK' '$GPU' '$CONFIG_PATH' '$EPOCHS' '$ENV_ID' '$NAME'" C-m
done

echo "Started tmux session: $SESSION_NAME (group=$GROUP, task=$TASK)"
