#!/usr/bin/env bash

# Two-prong followup queue:
#   1. slide W2 grid16 jvdirect seed sweep (s1, s2) — confirm 0.65 is reproducible
#   2. jp_absolute action repr on slide / insert / scoop with W2 grid16 backbone
#      — ceiling sweep shows jp_abs has highest replay-recoverable SR for
#        insert (0.8) and scoop (0.9), but never trained on it.
#
# Layout: 4 GPU workers, 5 runs total
#   gpu0: slide_w2_jvdirect_s1     -> slide_jpabs_s0
#   gpu1: slide_w2_jvdirect_s2
#   gpu2: insert_jpabs_s0
#   gpu3: scoop_jpabs_s0
#
# Waits for upstream queues (paperfaithful, relative_action, action_ablation,
# recommended_followup, overnight) to drain before launching.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/jpabs_seedsweep_20260504"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/jpabs_seedsweep_20260504"
LOG_ROOT="$ARTIFACT_ROOT/logs/jpabs_seedsweep_20260504"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/cpu_limit_env.sh"
WAIT_SESSION="${JPABS_WAIT_SESSION:-jpabs_seedsweep_wait}"
RUN_SESSION="${JPABS_RUN_SESSION:-jpabs_seedsweep_20260504}"
POLL_SEC="${POLL_SEC:-300}"
EPOCHS="${JPABS_EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-$ELSA_DATALOADER_WORKERS}"

CONFIG_SLIDE_JVDIRECT_W2="experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w2_grid16.yaml"
CONFIG_SLIDE_JPABS_W2="experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16.yaml"
CONFIG_INSERT_JPABS_W2="experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16.yaml"
CONFIG_SCOOP_JPABS_W2="experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16.yaml"

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

blockers_running() {
  pgrep -f "scripts/train_same_env_bcpolicy_probe.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_overnight_queue_pending_tmux.sh --worker" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/wait_and_run_volumedp_paperclose.sh" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_recommended_followup_queue_tmux.sh" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_paperfaithful_queue_tmux.sh" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_relative_action_4task_queue_tmux.sh" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_action_ablation_queue_tmux.sh" >/dev/null 2>&1 && return 0
  return 1
}

result_exists() {
  local task="$1"
  local run_name="$2"
  compgen -G "$RESULT_ROOT/$task/$run_name/*/result.json" >/dev/null
}

run_train() {
  local gpu="$1"
  local task="$2"
  local cfg="$3"
  local seed="$4"
  local run_name="$5"

  if result_exists "$task" "$run_name"; then
    echo "skip existing result: $run_name"
    return 0
  fi
  elsa_wait_for_existing_run "$run_name"
  if result_exists "$task" "$run_name"; then
    echo "skip existing result after wait: $run_name"
    return 0
  fi

  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "${ELSA_ENV_NAME:-elsa_challenge}"

  local python_bin="/home/cvlab-dgx/anaconda3/envs/${ELSA_ENV_NAME:-elsa_challenge}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  echo "=== START $run_name gpu=$gpu task=$task e$EPOCHS seed=$seed $(date '+%F %T') ==="
  set +e
  export CUDA_VISIBLE_DEVICES="$gpu"
  elsa_run_with_cpu_limit "$gpu" "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py" \
    --task "$task" \
    --dataset-config-path "$REPO_ROOT/$cfg" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --eval-episodes "$EVAL_EPISODES" \
    --device cuda:0 \
    --seed "$seed" \
    --run-name "$run_name" \
    --output-root "$RESULT_ROOT" \
    --checkpoint-root "$CKPT_ROOT" \
    2>&1 | tee "$LOG_ROOT/${run_name}.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$run_name exit=$status" | tee -a "$LOG_ROOT/_jpabs_status_gpu${gpu}.log"
  echo "=== END $run_name exit=$status $(date '+%F %T') ==="
  # Workers run in parallel; do not globally clean simulators here.
  sleep 5
  return "$status"
}

run_worker() {
  local gpu="$1"
  cd "$REPO_ROOT"

  case "$gpu" in
    0)
      run_train 0 slide_block_to_target "$CONFIG_SLIDE_JVDIRECT_W2" 1 \
        "slide_w2_jvdirect_grid16_e${EPOCHS}_s1"
      run_train 0 slide_block_to_target "$CONFIG_SLIDE_JPABS_W2" 0 \
        "slide_jpabs_w2_grid16_e${EPOCHS}_s0"
      ;;
    1)
      run_train 1 slide_block_to_target "$CONFIG_SLIDE_JVDIRECT_W2" 2 \
        "slide_w2_jvdirect_grid16_e${EPOCHS}_s2"
      ;;
    2)
      run_train 2 insert_onto_square_peg "$CONFIG_INSERT_JPABS_W2" 0 \
        "insert_jpabs_w2_grid16_e${EPOCHS}_s0"
      ;;
    3)
      run_train 3 scoop_with_spatula "$CONFIG_SCOOP_JPABS_W2" 0 \
        "scoop_jpabs_w2_grid16_e${EPOCHS}_s0"
      ;;
    *)
      echo "unknown gpu: $gpu" >&2
      return 2
      ;;
  esac
}

launch_run_session() {
  tmux has-session -t "$RUN_SESSION" 2>/dev/null && {
    echo "run session already exists: $RUN_SESSION"
    return 0
  }

  tmux new-session -d -s "$RUN_SESSION" -n gpu0_slide_chain
  tmux new-window -t "$RUN_SESSION" -n gpu1_slide_s2
  tmux new-window -t "$RUN_SESSION" -n gpu2_insert_jpabs
  tmux new-window -t "$RUN_SESSION" -n gpu3_scoop_jpabs

  local gpu
  for gpu in 0 1 2 3; do
    tmux send-keys -t "$RUN_SESSION:$gpu" \
      "cd '$REPO_ROOT' && bash '$0' --worker '$gpu'" C-m
  done

  echo "started run session: $RUN_SESSION"
  tmux list-windows -t "$RUN_SESSION"
}

run_waiter() {
  cd "$REPO_ROOT"
  local log="$LOG_ROOT/_launcher.log"
  exec > >(tee -a "$log") 2>&1

  echo "=== jpabs+seedsweep waiter start $(date '+%F %T') ==="
  while blockers_running; do
    echo "[$(date '+%F %T')] upstream training/queues still running; sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done

  echo "[$(date '+%F %T')] blockers clear; launching $RUN_SESSION"
  launch_run_session
}

launch_wait_session() {
  tmux has-session -t "$WAIT_SESSION" 2>/dev/null && {
    echo "wait session already exists: $WAIT_SESSION"
    return 0
  }
  tmux new-session -d -s "$WAIT_SESSION" -n launcher \
    "cd '$REPO_ROOT' && bash '$0' --waiter"
  echo "started wait session: $WAIT_SESSION"
  echo "logs: $LOG_ROOT/_launcher.log"
}

main() {
  case "${1:-}" in
    --waiter)
      run_waiter
      ;;
    --worker)
      run_worker "${2:?usage: $0 --worker <gpu>}"
      ;;
    *)
      launch_wait_session
      ;;
  esac
}

main "$@"
