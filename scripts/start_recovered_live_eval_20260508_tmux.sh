#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/recovered_live_eval_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/recovered_live_eval_20260508"
SESSION_NAME="${RECOVERED_EVAL_SESSION:-recovered_live_eval_20260508}"
EPISODES="${RECOVERED_EVAL_EPISODES:-20}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

run_eval() {
  local gpu="$1"
  local task="$2"
  local run_name="$3"
  local ckpt="$4"
  local cfg="$5"
  local mode="$6"

  local out="$RESULT_ROOT/$task/$run_name/$mode/result.json"
  if [ -f "$out" ]; then
    echo "skip existing eval: $out"
    return 0
  fi

  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"

  local python_bin="/home/cvlab-dgx/anaconda3/envs/${ENV_NAME}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/eval_flower_checkpoint_live.py"
    --model-path "$ckpt"
    --task "$task"
    --dataset-config-path "$cfg"
    --split training
    --env-ids 0
    --episodes "$EPISODES"
    --device cuda:0
    --output "$out"
  )
  if [ "$mode" = "hysteresis" ]; then
    cmd+=(
      --gripper-eval-mode hysteresis
      --gripper-open-threshold 0.65
      --gripper-close-threshold 0.35
      --gripper-min-hold-steps 2
    )
  fi

  echo "=== EVAL START gpu=$gpu task=$task run=$run_name mode=$mode episodes=$EPISODES $(date '+%F %T') ==="
  export CUDA_VISIBLE_DEVICES="$gpu"
  "${cmd[@]}"
  echo "=== EVAL END gpu=$gpu task=$task run=$run_name mode=$mode $(date '+%F %T') ==="
}

worker() {
  local gpu="$1"
  cd "$REPO_ROOT"
  case "$gpu" in
    0)
      run_eval 0 slide_block_to_target slide_jprel_w2_direct_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/slide_block_to_target/slide_jprel_w2_direct_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/slide_block_to_target/slide_jprel_w2_direct_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        threshold
      run_eval 0 close_box close_jprel_w4_jvservo_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/close_box/close_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/close_box/close_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        threshold
      run_eval 0 close_box close_jprel_w4_jvservo_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/close_box/close_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/close_box/close_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        hysteresis
      ;;
    1)
      run_eval 1 insert_onto_square_peg insert_jprel_w4_jvservo_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/insert_onto_square_peg/insert_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/insert_onto_square_peg/insert_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        threshold
      run_eval 1 insert_onto_square_peg insert_jprel_w4_jvservo_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/insert_onto_square_peg/insert_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/insert_onto_square_peg/insert_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        hysteresis
      run_eval 1 insert_onto_square_peg insert_jpabs_w2_grid16_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/jpabs_seedsweep_20260504/insert_onto_square_peg/insert_jpabs_w2_grid16_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/jpabs_seedsweep_20260504/insert_onto_square_peg/insert_jpabs_w2_grid16_e50_s0/env_000/resolved_config.yaml" \
        threshold
      ;;
    3)
      run_eval 3 scoop_with_spatula scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/scoop_with_spatula/scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/scoop_with_spatula/scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        threshold
      run_eval 3 scoop_with_spatula scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/action_ablation_20260504/scoop_with_spatula/scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/action_ablation_20260504/scoop_with_spatula/scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0/env_000/resolved_config.yaml" \
        hysteresis
      run_eval 3 scoop_with_spatula scoop_jpabs_w2_grid16_e50_s0 \
        "$ARTIFACT_ROOT/model_checkpoints/jpabs_seedsweep_20260504/scoop_with_spatula/scoop_jpabs_w2_grid16_e50_s0/env_000.pth" \
        "$ARTIFACT_ROOT/results/jpabs_seedsweep_20260504/scoop_with_spatula/scoop_jpabs_w2_grid16_e50_s0/env_000/resolved_config.yaml" \
        threshold
      ;;
    *)
      echo "unsupported worker gpu: $gpu" >&2
      return 2
      ;;
  esac
}

launch() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }

  tmux new-session -d -s "$SESSION_NAME" -n gpu0 \
    "cd '$REPO_ROOT' && bash '$0' --worker 0 >> '$LOG_ROOT/gpu0.log' 2>&1"
  tmux new-window -t "$SESSION_NAME" -n gpu1 \
    "cd '$REPO_ROOT' && bash '$0' --worker 1 >> '$LOG_ROOT/gpu1.log' 2>&1"
  tmux new-window -t "$SESSION_NAME" -n gpu3 \
    "cd '$REPO_ROOT' && bash '$0' --worker 3 >> '$LOG_ROOT/gpu3.log' 2>&1"

  echo "started session: $SESSION_NAME"
  echo "logs: $LOG_ROOT"
  tmux list-windows -t "$SESSION_NAME"
}

case "${1:-}" in
  --worker)
    worker "${2:?usage: $0 --worker <gpu>}"
    ;;
  *)
    launch
    ;;
esac

