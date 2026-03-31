#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ELSA_STORAGE_ROOT:-/mnt/ext_sdb1/elsa_robotics_challenge}/logs"
LOG_PATH="${ELSA_QUEUE_LOG:-$LOG_DIR/run_remaining_tasks_after_live_eval_$(date +%Y%m%d_%H%M%S).log}"

TASKS=(
  "insert_onto_square_peg"
  "scoop_with_spatula"
)

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_PATH"
}

wait_for_live_eval() {
  while tmux ls 2>/dev/null | rg -q '^elsa-live-sr-bc:'; do
    log "waiting for elsa-live-sr-bc to finish"
    sleep 60
  done
}

run_task() {
  local task="$1"
  log "starting download for task=${task}"
  (
    cd "$ROOT_DIR"
    export ELSA_TASK="$task"
    ./scripts/download_task_training.sh
  ) |& tee -a "$LOG_PATH"

  log "starting training for task=${task}"
  (
    cd "$ROOT_DIR"
    export ELSA_TASK="$task"
    export ELSA_NUM_SERVER_ROUNDS="${ELSA_NUM_SERVER_ROUNDS:-30}"
    export ELSA_LOCAL_EPOCHS="${ELSA_LOCAL_EPOCHS:-50}"
    export ELSA_FRACTION_FIT="${ELSA_FRACTION_FIT:-0.05}"
    export ELSA_FRACTION_EVAL="${ELSA_FRACTION_EVAL:-0.0025}"
    export ELSA_CLIENT_NUM_GPUS="${ELSA_CLIENT_NUM_GPUS:-0.188}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
    ./scripts/run_task_3gpu.sh
  ) |& tee -a "$LOG_PATH"

  log "finished task=${task}"
}

wait_for_live_eval
for task in "${TASKS[@]}"; do
  run_task "$task"
done
log "all remaining tasks finished"
