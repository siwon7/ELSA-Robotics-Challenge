#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${1:-same_env_diffusion_lora_ablation}"
TASK="slide_block_to_target"
ENV_ID="${ENV_ID:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="$ARTIFACT_ROOT/logs/same_env_diffusion_lora_ablation"
mkdir -p "$LOG_ROOT"

declare -a GPUS=("0" "1" "2" "3")
declare -a CONFIGS=(
  "experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jvdirect.yaml"
  "experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora8_jvdirect.yaml"
  "experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jpdirect_chunk4exec2.yaml"
  "experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora8_jpdirect_chunk4exec2.yaml"
)
declare -a RUN_PREFIXES=(
  "slide_sameenv_dinov3_diffusion_lora4_jvdirect"
  "slide_sameenv_dinov3_diffusion_lora8_jvdirect"
  "slide_sameenv_dinov3_diffusion_lora4_jpdirect_chunk4exec2"
  "slide_sameenv_dinov3_diffusion_lora8_jpdirect_chunk4exec2"
)

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "gpu0"

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  config="${CONFIGS[$idx]}"
  run_prefix="${RUN_PREFIXES[$idx]}"
  log_path="$LOG_ROOT/${run_prefix}.log"
  window_name="gpu${gpu}"

  if [ "$idx" -gt 0 ]; then
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
  fi

  cmd="cd '$REPO_ROOT' && bash '$SCRIPT_DIR/run_same_env_config_one_task.sh' '$TASK' '$gpu' '$config' '$EPOCHS' '$ENV_ID' '${run_prefix}_e${EPOCHS}_s${SEED}' '$EVAL_EPISODES' '$SEED' 2>&1 | tee '$log_path'"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" C-m
done

echo "started tmux session: $SESSION_NAME"
echo "logs: $LOG_ROOT"
