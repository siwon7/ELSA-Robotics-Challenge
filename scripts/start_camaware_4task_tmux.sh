#!/usr/bin/env bash
# Camera-aware 4-task baseline launch.
# Adds K (3x3) + T (4x4) flattened to low_dim_state (8 -> 33).
# Models: slide/close = JV-direct + gated FiLM; insert/scoop = JP-servo.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <session_name> [run_suffix]" >&2
  exit 1
fi

SESSION_NAME="$1"
RUN_SUFFIX="${2:-camaware_v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"

declare -a WINDOW_NAMES=(slide close_box insert scoop)
declare -a TASKS=(slide_block_to_target close_box insert_onto_square_peg scoop_with_spatula)
declare -a CONFIGS=(
  "experiments/slide_camaware_dino_depth_diffusion_lora8.yaml"
  "experiments/close_box_camaware_dino_depth_diffusion_lora8.yaml"
  "experiments/insert_camaware_dino_depth_diffusion_lora8.yaml"
  "experiments/scoop_camaware_dino_depth_diffusion_lora8.yaml"
)
declare -a GPUS=(0 1 2 3)

for cfg in "${CONFIGS[@]}"; do
  [ -f "$REPO_ROOT/$cfg" ] || { echo "missing $cfg" >&2; exit 1; }
done

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "${WINDOW_NAMES[0]}"

for idx in "${!TASKS[@]}"; do
  win="${WINDOW_NAMES[$idx]}"
  task="${TASKS[$idx]}"
  cfg="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  run_name="${task}_${RUN_SUFFIX}"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION_NAME" -n "$win"
  fi

  cmd="cd '$REPO_ROOT' && unset VIRTUAL_ENV && PATH='/home/cvlab-dgx/anaconda3/condabin:/usr/bin:/bin' && source '$SCRIPT_DIR/prepare_live_eval_env.sh' && source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh && conda activate '$ENV_NAME' && export ELSA_INCLUDE_CAMERA_IN_STATE=1 && CUDA_VISIBLE_DEVICES='$gpu' python scripts/train_same_env_bcpolicy_probe.py --task '$task' --dataset-config-path '$cfg' --epochs '$EPOCHS' --eval-episodes '$EVAL_EPISODES' --device cuda:0 --seed '$SEED' --run-name '$run_name' 2>&1 | tee 'logs/${run_name}.log'"
  tmux send-keys -t "$SESSION_NAME:$win" "$cmd" C-m
done

echo "started tmux session: $SESSION_NAME, suffix: $RUN_SUFFIX"
