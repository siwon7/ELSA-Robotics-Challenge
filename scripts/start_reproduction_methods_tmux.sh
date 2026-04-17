#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-repro_methods_current_repo}"
NUM_DEMOS="${2:-3}"
ENV_IDS="${3:-0,1}"
SESSION_NAME="${4:-repro_methods_current_repo}"

REPO_ROOT="/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge"
cd "${REPO_ROOT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -d -s "${SESSION_NAME}" -n close_box
tmux send-keys -t "${SESSION_NAME}:close_box" "cd ${REPO_ROOT} && bash scripts/run_reproduction_methods_one_task.sh close_box 0 ${NUM_DEMOS} ${ENV_IDS} ${RUN_TAG}" C-m

tmux new-window -t "${SESSION_NAME}" -n insert_onto_square_peg
tmux send-keys -t "${SESSION_NAME}:insert_onto_square_peg" "cd ${REPO_ROOT} && bash scripts/run_reproduction_methods_one_task.sh insert_onto_square_peg 1 ${NUM_DEMOS} ${ENV_IDS} ${RUN_TAG}" C-m

tmux new-window -t "${SESSION_NAME}" -n slide_block_to_target
tmux send-keys -t "${SESSION_NAME}:slide_block_to_target" "cd ${REPO_ROOT} && bash scripts/run_reproduction_methods_one_task.sh slide_block_to_target 2 ${NUM_DEMOS} ${ENV_IDS} ${RUN_TAG}" C-m

tmux new-window -t "${SESSION_NAME}" -n scoop_with_spatula
tmux send-keys -t "${SESSION_NAME}:scoop_with_spatula" "cd ${REPO_ROOT} && bash scripts/run_reproduction_methods_one_task.sh scoop_with_spatula 3 ${NUM_DEMOS} ${ENV_IDS} ${RUN_TAG}" C-m

echo "session=${SESSION_NAME}"
echo "run_tag=${RUN_TAG}"
