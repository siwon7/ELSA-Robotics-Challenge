#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/overnight_queue"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/overnight_queue"
LOG_ROOT="$ARTIFACT_ROOT/logs/overnight_queue"
SESSION_NAME="${OVERNIGHT_QUEUE_SESSION:-overnight_pending_16}"
TRAIN_ENV_IDS=(0 1 2 3 4)
EVAL_ENV_IDS=(0 1 2 3 4)

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

specs=(
  "0|same|slide_block_to_target|experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml|100|slide_baseline_jv_e100"
  "0|multi|slide_block_to_target|experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_proprio_gated_globalfilm.yaml|10|slide_volumedp_w4_film_5env_e10"
  "0|same|slide_block_to_target|experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml|50|slide_volumedp_w4_eeaux_e50"
  "0|same|slide_block_to_target|experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_proprio_gated_globalfilm_eeaux.yaml|50|slide_volumedp_w4_film_eeaux_e50"
  "1|same|close_box|experiments/close_box_sameenv_dino_depth_diffusion_lora8_jvdirect_proprio_gated_globalfilm.yaml|50|close_baseline_jv_propriofilm_e50"
  "1|multi|close_box|experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_proprio_gated_globalfilm.yaml|10|close_volumedp_w4_film_5env_e10"
  "1|same|close_box|experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml|50|close_volumedp_w4_eeaux_e50"
  "1|multi|close_box|experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml|10|close_volumedp_w4_eeaux_5env_e10"
  "2|multi|insert_onto_square_peg|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|10|insert_baseline_chunk4exec2_5env_e10"
  "2|same|insert_onto_square_peg|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|100|insert_baseline_chunk4exec2_e100"
  "2|same|insert_onto_square_peg|experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_w4_tight_grid16_eeaux.yaml|50|insert_volumedp_w4_eeaux_e50"
  "2|multi|insert_onto_square_peg|experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_w4_tight_grid16_eeaux.yaml|10|insert_volumedp_w4_eeaux_5env_e10"
  "3|multi|scoop_with_spatula|experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|10|scoop_baseline_chunk4exec2_5env_e10"
  "3|same|scoop_with_spatula|experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|100|scoop_baseline_chunk4exec2_e100"
  "3|same|scoop_with_spatula|experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_w4_tight_grid16_eeaux.yaml|50|scoop_volumedp_w4_eeaux_e50"
  "3|multi|scoop_with_spatula|experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_w4_tight_grid16_eeaux.yaml|10|scoop_volumedp_w4_eeaux_5env_e10"
)

window_name() {
  case "$1" in
    0) echo "gpu0_slide" ;;
    1) echo "gpu1_close" ;;
    2) echo "gpu2_insert" ;;
    3) echo "gpu3_scoop" ;;
    *) echo "gpu$1" ;;
  esac
}

result_glob_for() {
  local task="$1"
  local run_name="$2"
  echo "$RESULT_ROOT/$task/$run_name/*/result.json"
}

run_worker() {
  local gpu="$1"

  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  conda activate "${ELSA_ENV_NAME:-elsa_challenge}"
  set +e +o pipefail

  local python_bin="$CONDA_BASE/envs/${ELSA_ENV_NAME:-elsa_challenge}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  local status_log="$LOG_ROOT/_queue_status_gpu${gpu}.log"
  touch "$status_log"

  echo "=== WORKER GPU $gpu START $(date '+%F %T') ==="

  local ran_any=0
  local spec task cfg epochs run_name kind
  for spec in "${specs[@]}"; do
    IFS='|' read -r spec_gpu kind task cfg epochs run_name <<< "$spec"
    if [ "$spec_gpu" != "$gpu" ]; then
      continue
    fi

    if compgen -G "$(result_glob_for "$task" "$run_name")" > /dev/null; then
      echo "$run_name skip=existing_result" | tee -a "$status_log"
      continue
    fi

    ran_any=1
    echo "=== START $run_name at $(date '+%F %T') ==="

    local cmd=(
      "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py"
      --task "$task"
      --dataset-config-path "$REPO_ROOT/$cfg"
      --epochs "$epochs"
      --eval-episodes 20
      --device cuda:0
      --seed 0
      --run-name "$run_name"
      --output-root "$RESULT_ROOT"
      --checkpoint-root "$CKPT_ROOT"
    )

    if [ "$kind" = "multi" ]; then
      cmd+=(--train-env-ids "${TRAIN_ENV_IDS[@]}")
      cmd+=(--eval-env-ids "${EVAL_ENV_IDS[@]}")
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/${run_name}.log"
    local status="${PIPESTATUS[0]}"
    echo "$run_name exit=$status" | tee -a "$status_log"
    echo "=== END $run_name exit=$status at $(date '+%F %T') ==="

    pgrep -f CoppeliaSim | xargs -r kill -9 2>/dev/null
    sleep 5
  done

  if [ "$ran_any" -eq 0 ]; then
    echo "gpu${gpu}: nothing to run"
  fi

  echo "=== WORKER GPU $gpu DONE $(date '+%F %T') ==="
}

launch_tmux() {
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

  tmux new-session -d -s "$SESSION_NAME" -n "$(window_name 0)"
  tmux new-window -t "$SESSION_NAME" -n "$(window_name 1)"
  tmux new-window -t "$SESSION_NAME" -n "$(window_name 2)"
  tmux new-window -t "$SESSION_NAME" -n "$(window_name 3)"

  local gpu
  for gpu in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME:$(window_name "$gpu")" \
      "cd '$REPO_ROOT' && bash '$0' --worker '$gpu'" C-m
  done

  echo "session: $SESSION_NAME"
  tmux list-windows -t "$SESSION_NAME"
  echo ""
  echo "Pending queue per GPU:"
  for gpu in 0 1 2 3; do
    echo "GPU $gpu:"
    local spec task cfg epochs run_name kind
    for spec in "${specs[@]}"; do
      IFS='|' read -r spec_gpu kind task cfg epochs run_name <<< "$spec"
      if [ "$spec_gpu" = "$gpu" ]; then
        echo "  $run_name ($kind, $task, e$epochs)"
      fi
    done
  done
}

main() {
  if [ "${1:-}" = "--worker" ]; then
    if [ -z "${2:-}" ]; then
      echo "usage: $0 --worker <gpu>" >&2
      exit 2
    fi
    run_worker "$2"
    return
  fi

  launch_tmux
}

main "$@"
