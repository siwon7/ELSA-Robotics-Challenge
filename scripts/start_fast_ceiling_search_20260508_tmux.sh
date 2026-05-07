#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/fast_ceiling_search_20260508"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/fast_ceiling_search_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/fast_ceiling_search_20260508"
SESSION_NAME="${FAST_CEILING_SESSION:-fast_ceiling_search_20260508}"
WAITER_SESSION="${FAST_CEILING_WAITER_SESSION:-fast_ceiling_search_wait_20260508}"
EPOCHS="${FAST_CEILING_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-1}"
POLL_SEC="${POLL_SEC:-300}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

active_blockers() {
  tmux has-session -t ralph_fill4_power_moved_20260507 2>/dev/null && return 0
  tmux has-session -t recovered_live_eval_20260508 2>/dev/null && return 0
  tmux has-session -t insert_volumedp_recovery_20260508 2>/dev/null && return 0
  tmux has-session -t demo_retrieval_probe_wait_20260508 2>/dev/null && return 0
  tmux has-session -t demo_retrieval_probe_20260508 2>/dev/null && return 0
  tmux has-session -t demo_action_sweep_20260508 2>/dev/null && return 0
  tmux has-session -t demo_action_sweep_followup_20260508 2>/dev/null && return 0
  tmux has-session -t demo_action_sweep_full_20260508 2>/dev/null && return 0
  tmux has-session -t demo_action_sweep_train_long_20260508 2>/dev/null && return 0
  pgrep -f "scripts/train_same_env_bcpolicy_probe.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/eval_flower_checkpoint_live.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/eval_demo_retrieval_policy_live.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/start_demo_action_sweep_followup_20260508_tmux.sh" >/dev/null 2>&1 && return 0
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
  source "$SCRIPT_DIR/cpu_limit_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"

  local python_bin="/home/cvlab-dgx/anaconda3/envs/${ENV_NAME}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py"
    --task "$task"
    --dataset-config-path "$REPO_ROOT/$cfg"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --eval-episodes 20
    --device cuda:0
    --seed 0
    --run-name "$run_name"
    --output-root "$RESULT_ROOT"
    --checkpoint-root "$CKPT_ROOT"
  )

  echo "=== FAST CEILING START gpu=$gpu task=$task run=$run_name cfg=$cfg epochs=$EPOCHS $(date '+%F %T') ==="
  export CUDA_VISIBLE_DEVICES="$gpu"
  elsa_run_with_cpu_limit "$gpu" "${cmd[@]}"
  echo "=== FAST CEILING END gpu=$gpu task=$task run=$run_name $(date '+%F %T') ==="
}

worker() {
  local gpu="$1"
  cd "$REPO_ROOT"
  case "$gpu" in
    0)
      run_train 0 slide_block_to_target \
        experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux.yaml \
        slide_volumedp_jprel_w4_direct_grid16_eeaux_fastceil_e${EPOCHS}_s0
      ;;
    1)
      run_train 1 close_box \
        experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
        close_dinodepth_jvdirect_fastceil_e${EPOCHS}_s0
      ;;
    2)
      run_train 2 insert_onto_square_peg \
        experiments/insert_sameenv_dino_depth_diffusion_lora8_jpkeyframe4_jpservo.yaml \
        insert_dinodepth_jpkeyframe4_jpservo_fastceil_e${EPOCHS}_s0
      ;;
    3)
      run_train 3 scoop_with_spatula \
        experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
        scoop_dinodepth_jvdirect_fastceil_e${EPOCHS}_s0
      ;;
    *)
      echo "unsupported worker gpu: $gpu" >&2
      return 2
      ;;
  esac
}

launch_run_session() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }

  for gpu in 0 1 2 3; do
    if [ "$gpu" -eq 0 ]; then
      tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu}" \
        "cd '$REPO_ROOT' && bash '$0' --worker '$gpu' >> '$LOG_ROOT/gpu${gpu}.log' 2>&1"
    else
      tmux new-window -t "$SESSION_NAME" -n "gpu${gpu}" \
        "cd '$REPO_ROOT' && bash '$0' --worker '$gpu' >> '$LOG_ROOT/gpu${gpu}.log' 2>&1"
    fi
  done
  echo "started session: $SESSION_NAME"
  echo "logs: $LOG_ROOT"
  tmux list-windows -t "$SESSION_NAME"
}

waiter() {
  cd "$REPO_ROOT"
  local log="$LOG_ROOT/_waiter.log"
  exec > >(tee -a "$log") 2>&1
  echo "=== FAST CEILING WAITER START $(date '+%F %T') ==="
  while active_blockers; do
    echo "[$(date '+%F %T')] blockers active; sleeping ${POLL_SEC}s"
    tmux ls 2>/dev/null || true
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits || true
    sleep "$POLL_SEC"
  done
  echo "[$(date '+%F %T')] blockers clear; launching fast ceiling search"
  launch_run_session
}

launch_waiter() {
  tmux has-session -t "$WAITER_SESSION" 2>/dev/null && {
    echo "waiter session already exists: $WAITER_SESSION"
    return 0
  }
  tmux new-session -d -s "$WAITER_SESSION" -n waiter \
    "cd '$REPO_ROOT' && bash '$0' --waiter"
  echo "started waiter: $WAITER_SESSION"
  echo "logs: $LOG_ROOT/_waiter.log"
}

case "${1:-}" in
  --worker)
    worker "${2:?usage: $0 --worker <gpu>}"
    ;;
  --waiter)
    waiter
    ;;
  --launch-now)
    launch_run_session
    ;;
  *)
    launch_waiter
    ;;
esac
