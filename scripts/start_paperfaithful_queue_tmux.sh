#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/paperfaithful_20260504"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/paperfaithful_20260504"
LOG_ROOT="$ARTIFACT_ROOT/logs/paperfaithful_20260504"
WAIT_SESSION="${PAPERFAITHFUL_WAIT_SESSION:-paperfaithful_wait}"
RUN_SESSION="${PAPERFAITHFUL_RUN_SESSION:-paperfaithful_20260504}"
POLL_SEC="${POLL_SEC:-300}"
EPOCHS="${PAPERFAITHFUL_EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-0}"

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

blockers_running() {
  pgrep -f "scripts/start_overnight_queue_pending_tmux.sh --worker" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/wait_and_run_volumedp_paperclose.sh" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_recommended_followup_queue_tmux.sh" >/dev/null 2>&1 && return 0
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
  local run_name="$4"

  if result_exists "$task" "$run_name"; then
    echo "skip existing result: $run_name"
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

  echo "=== START $run_name gpu=$gpu task=$task e$EPOCHS seed=$SEED $(date '+%F %T') ==="
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py" \
    --task "$task" \
    --dataset-config-path "$REPO_ROOT/$cfg" \
    --epochs "$EPOCHS" \
    --eval-episodes "$EVAL_EPISODES" \
    --device cuda:0 \
    --seed "$SEED" \
    --run-name "$run_name" \
    --output-root "$RESULT_ROOT" \
    --checkpoint-root "$CKPT_ROOT" \
    2>&1 | tee "$LOG_ROOT/${run_name}.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$run_name exit=$status" | tee -a "$LOG_ROOT/_paperfaithful_status_gpu${gpu}.log"
  echo "=== END $run_name exit=$status $(date '+%F %T') ==="
  pgrep -f CoppeliaSim | xargs -r kill -9 2>/dev/null || true
  sleep 5
  return "$status"
}

run_worker() {
  local gpu="$1"
  cd "$REPO_ROOT"

  case "$gpu" in
    0)
      run_train 0 slide_block_to_target \
        experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_paperfaithful_v2.yaml \
        slide_volumedp_jpservo_paperfaithful_v2_e${EPOCHS}_s${SEED}
      ;;
    1)
      run_train 1 close_box \
        experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_paperfaithful_v2.yaml \
        close_volumedp_jpservo_paperfaithful_v2_e${EPOCHS}_s${SEED}
      ;;
    2)
      run_train 2 insert_onto_square_peg \
        experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_paperfaithful_v2.yaml \
        insert_volumedp_jpservo_paperfaithful_v2_e${EPOCHS}_s${SEED}
      ;;
    3)
      run_train 3 scoop_with_spatula \
        experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpservo_paperfaithful_v2.yaml \
        scoop_volumedp_jpservo_paperfaithful_v2_e${EPOCHS}_s${SEED}
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

  tmux new-session -d -s "$RUN_SESSION" -n gpu0_slide
  tmux new-window -t "$RUN_SESSION" -n gpu1_close
  tmux new-window -t "$RUN_SESSION" -n gpu2_insert
  tmux new-window -t "$RUN_SESSION" -n gpu3_scoop

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

  echo "=== paperfaithful waiter start $(date '+%F %T') ==="
  while blockers_running; do
    echo "[$(date '+%F %T')] upstream queues still running; sleeping ${POLL_SEC}s"
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
