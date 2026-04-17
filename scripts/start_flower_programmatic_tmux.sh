#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-flower_abs_joint_pos_prog_v1}"
RUN_TAG="${RUN_TAG:-abs-jpos-prog-v1}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
FRACTION_FIT="${FRACTION_FIT:-0.05}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-1.0}"
CLIENT_NUM_GPUS="${CLIENT_NUM_GPUS:-0.25}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME" -n close_box
tmux send-keys -t "$SESSION_NAME:close_box" \
  "cd '$REPO_ROOT' && CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' close_box 0 '$NUM_ROUNDS' '$LOCAL_EPOCHS' '$RUN_TAG' '$FRACTION_FIT' '$TRAIN_SPLIT'" C-m

tmux new-window -t "$SESSION_NAME" -n slide_block_to_target
tmux send-keys -t "$SESSION_NAME:slide_block_to_target" \
  "cd '$REPO_ROOT' && CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' slide_block_to_target 1 '$NUM_ROUNDS' '$LOCAL_EPOCHS' '$RUN_TAG' '$FRACTION_FIT' '$TRAIN_SPLIT'" C-m

tmux new-window -t "$SESSION_NAME" -n insert_onto_square_peg
tmux send-keys -t "$SESSION_NAME:insert_onto_square_peg" \
  "cd '$REPO_ROOT' && CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' insert_onto_square_peg 2 '$NUM_ROUNDS' '$LOCAL_EPOCHS' '$RUN_TAG' '$FRACTION_FIT' '$TRAIN_SPLIT'" C-m

tmux new-window -t "$SESSION_NAME" -n scoop_with_spatula
tmux send-keys -t "$SESSION_NAME:scoop_with_spatula" \
  "cd '$REPO_ROOT' && CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' scoop_with_spatula 3 '$NUM_ROUNDS' '$LOCAL_EPOCHS' '$RUN_TAG' '$FRACTION_FIT' '$TRAIN_SPLIT'" C-m

tmux new-window -t "$SESSION_NAME" -n monitor
tmux send-keys -t "$SESSION_NAME:monitor" \
  "cd '$REPO_ROOT' && MONITOR_INTERVAL_SEC=1800 bash '$SCRIPT_DIR/monitor_flower_programmatic.sh' '$SESSION_NAME' '$RUN_TAG'" C-m

echo "Started tmux session: $SESSION_NAME"
