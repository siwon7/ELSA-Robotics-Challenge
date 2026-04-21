#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"
RESULT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}/results/same_env_suite"

CONFIG_JV="experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jvdirect.yaml"
CONFIG_JP_ONESTEP="experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jpdirect_onestep.yaml"
CONFIG_JP_CHUNK="experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora8_jpdirect_chunk4exec2.yaml"
CONFIG_JP_KEYFRAME="experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jpkeyframe4.yaml"

cd "$REPO_ROOT"

wait_for_wave() {
  local prefix="$1"
  local tasks=("close_box" "insert_onto_square_peg" "scoop_with_spatula" "slide_block_to_target")
  while true; do
    local done=1
    for task in "${tasks[@]}"; do
      local path="$RESULT_ROOT/$task/${task}_${prefix}_e${EPOCHS}_s${SEED}/env_000/result.json"
      if [ ! -f "$path" ]; then
        done=0
        break
      fi
    done
    if [ "$done" -eq 1 ]; then
      break
    fi
    sleep 60
  done
}

echo "[wave 1] JV direct"
EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  bash "$SCRIPT_DIR/start_same_env_multitask_config_tmux.sh" \
  "$CONFIG_JV" \
  "dinov3_diffusion_lora4_jvdirect_mt" \
  "same_env_multitask_dinov3_diffusion_lora4_jvdirect_mt"
wait_for_wave "dinov3_diffusion_lora4_jvdirect_mt"

echo "[wave 2] JP one-step"
EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  bash "$SCRIPT_DIR/start_same_env_multitask_config_tmux.sh" \
  "$CONFIG_JP_ONESTEP" \
  "dinov3_diffusion_lora4_jpdirect_onestep_mt" \
  "same_env_multitask_dinov3_diffusion_lora4_jpdirect_onestep_mt"
wait_for_wave "dinov3_diffusion_lora4_jpdirect_onestep_mt"

echo "[wave 3] JP chunk4+exec2"
EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  bash "$SCRIPT_DIR/start_same_env_multitask_config_tmux.sh" \
  "$CONFIG_JP_CHUNK" \
  "dinov3_diffusion_lora8_jpdirect_chunk4exec2_mt" \
  "same_env_multitask_dinov3_diffusion_lora8_jpdirect_chunk4exec2_mt"
wait_for_wave "dinov3_diffusion_lora8_jpdirect_chunk4exec2_mt"

echo "[wave 4] JP keyframe4"
EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  bash "$SCRIPT_DIR/start_same_env_multitask_config_tmux.sh" \
  "$CONFIG_JP_KEYFRAME" \
  "dinov3_diffusion_lora4_jpkeyframe4_mt" \
  "same_env_multitask_dinov3_diffusion_lora4_jpkeyframe4_mt"
wait_for_wave "dinov3_diffusion_lora4_jpkeyframe4_mt"

echo "completed 4-wave VolumeDP-style action sweep"
