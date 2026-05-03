#!/usr/bin/env bash
# Eval-only rerun for VolumeDP-full smoke checkpoints.
# Training crashed at post-train eval because live_rollout did not pass
# obs_context to the encoder for vision_backbone=volumedp_full_dinov3_depth.
# The checkpoints are intact; this script reruns online_evaluation with the
# fix in live_rollout.py (startswith("volumedp_") covers both lite and full).
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <session_name> [eval_episodes]" >&2
  exit 1
fi

SESSION_NAME="$1"
EVAL_EPISODES="${2:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

ARTIFACT_ROOT="/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/volumedp_full_smoke"
OUT_ROOT="$ARTIFACT_ROOT/results/volumedp_full_smoke"
LOG_DIR="$ARTIFACT_ROOT/logs/volumedp_full_smoke"
mkdir -p "$LOG_DIR"

declare -a WINDOW_NAMES=(slide close_box insert scoop)
declare -a TASKS=(slide_block_to_target close_box insert_onto_square_peg scoop_with_spatula)
declare -a CONFIGS=(
  "experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect.yaml"
  "experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect.yaml"
  "experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpservo.yaml"
  "experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpservo.yaml"
)
declare -a RUN_NAMES=(
  "slide_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_smoke_v1"
  "close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_smoke_v1"
  "insert_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_smoke_v1"
  "scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_smoke_v1"
)
declare -a GPUS=(0 1 2 3)

for cfg in "${CONFIGS[@]}"; do
  [ -f "$REPO_ROOT/$cfg" ] || { echo "missing $cfg" >&2; exit 1; }
done
for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  run="${RUN_NAMES[$idx]}"
  ckpt="$CKPT_ROOT/$task/$run/env_000.pth"
  [ -f "$ckpt" ] || { echo "missing checkpoint: $ckpt" >&2; exit 1; }
done

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "${WINDOW_NAMES[0]}"

for idx in "${!TASKS[@]}"; do
  win="${WINDOW_NAMES[$idx]}"
  task="${TASKS[$idx]}"
  cfg="${CONFIGS[$idx]}"
  run="${RUN_NAMES[$idx]}"
  gpu="${GPUS[$idx]}"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION_NAME" -n "$win"
  fi

  cmd="cd '$REPO_ROOT' && unset VIRTUAL_ENV && PATH='/home/cvlab-dgx/anaconda3/condabin:/usr/bin:/bin' && source '$SCRIPT_DIR/prepare_live_eval_env.sh' && source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh && conda activate '$ENV_NAME' && CUDA_VISIBLE_DEVICES='$gpu' python scripts/eval_camaware_checkpoint.py --task '$task' --dataset-config-path '$cfg' --run-name '$run' --checkpoint-root '$CKPT_ROOT' --output-root '$OUT_ROOT' --eval-episodes '$EVAL_EPISODES' --device cuda:0 2>&1 | tee '$LOG_DIR/${run}_evalonly.log'"
  tmux send-keys -t "$SESSION_NAME:$win" "$cmd" C-m
done

echo "started tmux session: $SESSION_NAME"
echo "logs: $LOG_DIR/*_evalonly.log"
echo "results: $OUT_ROOT/<task>/<run>/env_000/result.json"
