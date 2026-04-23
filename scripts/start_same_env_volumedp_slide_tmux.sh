#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge"
PYTHON_BIN="/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python"
SESSION_NAME="${1:-same_env_volumedp_slide_v1}"
TASK="slide_block_to_target"

declare -a GPUS=(0 1 2 3)
declare -a CONFIGS=(
  "$ROOT_DIR/experiments/slide_block_to_target_sameenv_volumedp_lite_dinov3_jvdirect_lora4_v1.yaml"
  "$ROOT_DIR/experiments/slide_block_to_target_sameenv_volumedp_lite_dinov3_jvdirect_lora8_v1.yaml"
  "$ROOT_DIR/experiments/slide_block_to_target_sameenv_volumedp_lite_dinov3_jvdirect_lora8_deep_v1.yaml"
  "$ROOT_DIR/experiments/slide_block_to_target_sameenv_volumedp_lite_dinov3_jvdirect_lora8_deep_grid10_v1.yaml"
)
declare -a RUN_NAMES=(
  "slide_sameenv_volumedp_lite_dinov3_jv_lora4_v1"
  "slide_sameenv_volumedp_lite_dinov3_jv_lora8_v1"
  "slide_sameenv_volumedp_lite_dinov3_jv_lora8_deep_v1"
  "slide_sameenv_volumedp_lite_dinov3_jv_lora8_deep_grid10_v1"
)

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "${RUN_NAMES[0]}"

for idx in "${!CONFIGS[@]}"; do
  if [[ "$idx" -gt 0 ]]; then
    tmux new-window -t "$SESSION_NAME" -n "${RUN_NAMES[$idx]}"
  fi
  cmd="source $ROOT_DIR/scripts/prepare_live_eval_env.sh && \
export PYTHONPATH=$ROOT_DIR:/tmp/robot-colosseum && \
CUDA_VISIBLE_DEVICES=${GPUS[$idx]} $PYTHON_BIN $ROOT_DIR/scripts/train_same_env_bcpolicy_probe.py \
  --task $TASK \
  --dataset-config-path ${CONFIGS[$idx]} \
  --env-id 0 \
  --epochs 50 \
  --train-split 0.9 \
  --batch-size 32 \
  --num-workers 4 \
  --eval-episodes 20 \
  --device cuda:0 \
  --run-name ${RUN_NAMES[$idx]} \
  --output-root /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite \
  --checkpoint-root /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints/same_env_suite \
  --seed 0 |& tee /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/same_env_suite/${RUN_NAMES[$idx]}.log"
  tmux send-keys -t "$SESSION_NAME:$idx" "$cmd" C-m
done

echo "Started session: $SESSION_NAME"
