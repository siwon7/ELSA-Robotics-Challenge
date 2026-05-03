#!/usr/bin/env bash
# Run one VolumeDP-full ablation wave: train on 4 GPUs in parallel for 4 tasks,
# then run eval on the resulting checkpoints. Each wave is identified by a
# suffix used for both run_name and config filename pattern.
#
# Usage: run_volumedp_full_wave.sh <wave_suffix>
#   wave_suffix in {w1_global, w2_grid16, w3_tightbnds}
#
# Train + eval are launched serially in 4 tmux windows (one per task), each
# pinned to a single GPU. The script blocks until all 4 windows complete.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <wave_suffix>" >&2
  exit 1
fi

WAVE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"

ARTIFACT_ROOT="/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/volumedp_full_ablations"
OUT_ROOT="$ARTIFACT_ROOT/results/volumedp_full_ablations"
LOG_DIR="$ARTIFACT_ROOT/logs/volumedp_full_ablations"
mkdir -p "$LOG_DIR"

declare -a WINDOW_NAMES=(slide close_box insert scoop)
declare -a TASKS=(slide_block_to_target close_box insert_onto_square_peg scoop_with_spatula)
declare -a CONFIGS=(
  "experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_${WAVE}.yaml"
  "experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_${WAVE}.yaml"
  "experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_${WAVE}.yaml"
  "experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_${WAVE}.yaml"
)
declare -a RUN_BASES=(
  "slide_sameenv_volumedp_full_${WAVE}"
  "close_box_sameenv_volumedp_full_${WAVE}"
  "insert_sameenv_volumedp_full_${WAVE}"
  "scoop_sameenv_volumedp_full_${WAVE}"
)
declare -a GPUS=(0 1 2 3)

for cfg in "${CONFIGS[@]}"; do
  [ -f "$REPO_ROOT/$cfg" ] || { echo "missing $cfg" >&2; exit 1; }
done

SESSION="vdp_${WAVE}"
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" -n "${WINDOW_NAMES[0]}"

for idx in "${!TASKS[@]}"; do
  win="${WINDOW_NAMES[$idx]}"
  task="${TASKS[$idx]}"
  cfg="${CONFIGS[$idx]}"
  run="${RUN_BASES[$idx]}"
  gpu="${GPUS[$idx]}"
  done_marker="$LOG_DIR/${run}.done"
  rm -f "$done_marker"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION" -n "$win"
  fi

  # Train then eval. Both write logs. Final touch creates the done marker
  # so the orchestrator can poll for completion.
  cmd="cd '$REPO_ROOT' && unset VIRTUAL_ENV && PATH='/home/cvlab-dgx/anaconda3/condabin:/usr/bin:/bin' && source '$SCRIPT_DIR/prepare_live_eval_env.sh' && source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh && conda activate '$ENV_NAME' && CUDA_VISIBLE_DEVICES='$gpu' python scripts/train_same_env_bcpolicy_probe.py --task '$task' --dataset-config-path '$cfg' --epochs '$EPOCHS' --eval-episodes '$EVAL_EPISODES' --device cuda:0 --seed '$SEED' --run-name '$run' --output-root '$OUT_ROOT' --checkpoint-root '$CKPT_ROOT' 2>&1 | tee '$LOG_DIR/${run}_train.log' ; touch '$done_marker'"
  tmux send-keys -t "$SESSION:$win" "$cmd" C-m
done

echo "started tmux session: $SESSION"
echo "logs: $LOG_DIR/*${WAVE}*"
echo "done markers: $LOG_DIR/*.done"
