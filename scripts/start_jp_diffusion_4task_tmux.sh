#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <session_name> [run_suffix]" >&2
  exit 1
fi

SESSION_NAME="$1"
RUN_SUFFIX="${2:-jp_diffusion_4task}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"

declare -a WINDOW_NAMES=(
  "slide"
  "close_box"
  "insert"
  "scoop"
)

declare -a TASKS=(
  "slide_block_to_target"
  "close_box"
  "insert_onto_square_peg"
  "scoop_with_spatula"
)

declare -a CONFIGS=(
  "experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml"
  "experiments/close_box_sameenv_dino_depth_diffusion_lora8_jvdirect_splitgripper_e100.yaml"
  "experiments/insert_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml"
  "experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml"
)

declare -a GPUS=("0" "1" "2" "3")

for config_path in "${CONFIGS[@]}"; do
  if [ ! -f "$REPO_ROOT/$config_path" ]; then
    echo "missing config: $config_path" >&2
    exit 1
  fi
done

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "${WINDOW_NAMES[0]}"

for idx in "${!TASKS[@]}"; do
  window_name="${WINDOW_NAMES[$idx]}"
  task="${TASKS[$idx]}"
  config_path="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  run_name="${task}_${RUN_SUFFIX}"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
  fi

  cmd="cd '$REPO_ROOT' && source '$SCRIPT_DIR/prepare_live_eval_env.sh' && conda activate '$ENV_NAME' && CUDA_VISIBLE_DEVICES='$gpu' python scripts/train_same_env_bcpolicy_probe.py --task '$task' --dataset-config-path '$config_path' --epochs '$EPOCHS' --eval-episodes '$EVAL_EPISODES' --device cuda:0 --seed '$SEED' --run-name '$run_name'"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" C-m
done

echo "started tmux session: $SESSION_NAME"
echo "run suffix: $RUN_SUFFIX"
