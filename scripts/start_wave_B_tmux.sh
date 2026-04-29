#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$SCRIPT_DIR/start_long_pair_sweep_tmux.sh"

SESSION_NAME="phase1_waveB"
RUN_SUFFIX="phase1B"
GPU_CSV="0,1,2,3"

EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"
ENV_ID="${ENV_ID:-0}"

declare -a TASK_CONFIG_PAIRS=(
  "close_box:experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_g30c10_splitgripper.yaml"
  "close_box:experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_g20c05_splitgripper.yaml"
  "slide_block_to_target:experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_g30c10.yaml"
  "slide_block_to_target:experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_g20c05.yaml"
)

if [ ! -f "$LAUNCHER" ]; then
  echo "launcher script not found: $LAUNCHER" >&2
  exit 1
fi

for pair in "${TASK_CONFIG_PAIRS[@]}"; do
  config_rel="${pair#*:}"
  config_path="$REPO_ROOT/$config_rel"

  if [ ! -f "$config_path" ]; then
    echo "missing Wave B config: $config_path" >&2
    exit 1
  fi
done

EPOCHS="$EPOCHS" \
EVAL_EPISODES="$EVAL_EPISODES" \
SEED="$SEED" \
ENV_ID="$ENV_ID" \
bash "$LAUNCHER" \
  "$SESSION_NAME" \
  "$RUN_SUFFIX" \
  "$GPU_CSV" \
  "${TASK_CONFIG_PAIRS[@]}"

echo "Wave B launched: session=$SESSION_NAME run_suffix=$RUN_SUFFIX gpus=$GPU_CSV epochs=$EPOCHS eval_episodes=$EVAL_EPISODES seed=$SEED env_id=$ENV_ID"
echo "Reserved for a follow-up wave: experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2.yaml and experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml"
