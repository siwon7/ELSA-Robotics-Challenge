#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${1:-same_env_multi_env_best_v1}"
TRAIN_ENV_IDS="${TRAIN_ENV_IDS:-0,1,2,3,4}"
EVAL_ENV_IDS="${EVAL_ENV_IDS:-0,1,2,3,4}"
EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n slide_nofilm

tmux send-keys -t "$SESSION_NAME:slide_nofilm" \
  "cd '$REPO_ROOT' && bash scripts/run_same_env_multi_env_config_one_task.sh slide_block_to_target 0 experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml '$EPOCHS' '$TRAIN_ENV_IDS' '$EVAL_ENV_IDS' slide_multienv_dino_depth_lora8_jv_v1 '$EVAL_EPISODES' '$SEED'" C-m

tmux new-window -t "$SESSION_NAME" -n slide_gated
tmux send-keys -t "$SESSION_NAME:slide_gated" \
  "cd '$REPO_ROOT' && bash scripts/run_same_env_multi_env_config_one_task.sh slide_block_to_target 1 experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jvdirect_proprio_gated_globalfilm.yaml '$EPOCHS' '$TRAIN_ENV_IDS' '$EVAL_ENV_IDS' slide_multienv_dino_depth_lora8_jv_gatedfilm_v1 '$EVAL_EPISODES' '$SEED'" C-m

tmux new-window -t "$SESSION_NAME" -n close_nofilm
tmux send-keys -t "$SESSION_NAME:close_nofilm" \
  "cd '$REPO_ROOT' && bash scripts/run_same_env_multi_env_config_one_task.sh close_box 2 experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml '$EPOCHS' '$TRAIN_ENV_IDS' '$EVAL_ENV_IDS' close_multienv_dino_depth_lora8_jv_v1 '$EVAL_EPISODES' '$SEED'" C-m

tmux new-window -t "$SESSION_NAME" -n close_weakfilm
tmux send-keys -t "$SESSION_NAME:close_weakfilm" \
  "cd '$REPO_ROOT' && bash scripts/run_same_env_multi_env_config_one_task.sh close_box 3 experiments/close_box_sameenv_dino_depth_diffusion_lora8_jvdirect_proprio_globalfilm_weak01.yaml '$EPOCHS' '$TRAIN_ENV_IDS' '$EVAL_ENV_IDS' close_multienv_dino_depth_lora8_jv_weakfilm_v1 '$EVAL_EPISODES' '$SEED'" C-m

echo "$SESSION_NAME"
