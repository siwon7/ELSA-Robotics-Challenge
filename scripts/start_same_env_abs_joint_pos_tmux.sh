#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${1:-abs_joint_pos_same_env}"
RUN_NAME="${2:-bcpolicy_abs_joint_pos_e50_env0}"
EPOCHS="${3:-50}"
ENV_ID="${4:-0}"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -n close_box
tmux send-keys -t "$SESSION_NAME:close_box" \
  "cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_abs_joint_pos_one_task.sh' close_box 0 '$EPOCHS' '$ENV_ID' '$RUN_NAME'" C-m

tmux new-window -t "$SESSION_NAME" -n slide_block_to_target
tmux send-keys -t "$SESSION_NAME:slide_block_to_target" \
  "cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_abs_joint_pos_one_task.sh' slide_block_to_target 1 '$EPOCHS' '$ENV_ID' '$RUN_NAME'" C-m

tmux new-window -t "$SESSION_NAME" -n insert_onto_square_peg
tmux send-keys -t "$SESSION_NAME:insert_onto_square_peg" \
  "cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_abs_joint_pos_one_task.sh' insert_onto_square_peg 2 '$EPOCHS' '$ENV_ID' '$RUN_NAME'" C-m

tmux new-window -t "$SESSION_NAME" -n scoop_with_spatula
tmux send-keys -t "$SESSION_NAME:scoop_with_spatula" \
  "cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_abs_joint_pos_one_task.sh' scoop_with_spatula 3 '$EPOCHS' '$ENV_ID' '$RUN_NAME'" C-m

echo "Started tmux session: $SESSION_NAME"
