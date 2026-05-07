#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/demo_retrieval_probe_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/demo_retrieval_probe_20260508"
SESSION_NAME="${DEMO_RETRIEVAL_SESSION:-demo_retrieval_probe_20260508}"
WAITER_SESSION="${DEMO_RETRIEVAL_WAITER_SESSION:-demo_retrieval_probe_wait_20260508}"
EPISODES="${DEMO_RETRIEVAL_EPISODES:-20}"
MAX_STEPS="${DEMO_RETRIEVAL_MAX_STEPS:-300}"
POLL_SEC="${POLL_SEC:-300}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

active_blockers() {
  tmux has-session -t ralph_fill4_power_moved_20260507 2>/dev/null && return 0
  tmux has-session -t recovered_live_eval_20260508 2>/dev/null && return 0
  tmux has-session -t insert_volumedp_recovery_20260508 2>/dev/null && return 0
  tmux has-session -t fast_ceiling_search_20260508 2>/dev/null && return 0
  pgrep -f "scripts/train_same_env_bcpolicy_probe.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/eval_flower_checkpoint_live.py" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/eval_demo_retrieval_policy_live.py" >/dev/null 2>&1 && return 0
  return 1
}

result_exists() {
  local task="$1"
  local run_name="$2"
  [ -s "$RESULT_ROOT/$task/$run_name/result.json" ]
}

run_eval() {
  local task="$1"
  local cfg="$2"
  local run_name="$3"
  local phase_weight="${4:-0.5}"
  local top_k="${5:-1}"

  if result_exists "$task" "$run_name"; then
    echo "skip existing result: $task/$run_name"
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

  local output="$RESULT_ROOT/$task/$run_name/result.json"
  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/eval_demo_retrieval_policy_live.py"
    --task "$task"
    --dataset-config-path "$REPO_ROOT/$cfg"
    --rollout-split training
    --env-ids 0
    --index-env-ids 0
    --index-split train
    --episodes "$EPISODES"
    --max-steps "$MAX_STEPS"
    --state-dim 8
    --phase-weight "$phase_weight"
    --top-k "$top_k"
    --output "$output"
  )

  echo "=== DEMO RETRIEVAL START task=$task run=$run_name cfg=$cfg episodes=$EPISODES $(date '+%F %T') ==="
  export CUDA_VISIBLE_DEVICES=""
  export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
  export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
  elsa_run_with_cpu_limit 0 "${cmd[@]}"
  echo "=== DEMO RETRIEVAL END task=$task run=$run_name $(date '+%F %T') ==="
}

run_sequence() {
  cd "$REPO_ROOT"
  run_eval slide_block_to_target \
    experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux.yaml \
    slide_statephase_jprel_w4_direct_trainenv0_e${EPISODES} \
    0.5 1
  run_eval close_box \
    experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
    close_statephase_jvdirect_trainenv0_e${EPISODES} \
    0.5 1
  run_eval insert_onto_square_peg \
    experiments/insert_sameenv_dino_depth_diffusion_lora8_jpkeyframe4_jpservo.yaml \
    insert_statephase_jpkeyframe4_jpservo_trainenv0_e${EPISODES} \
    0.5 1
  run_eval scoop_with_spatula \
    experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
    scoop_statephase_jvdirect_trainenv0_e${EPISODES} \
    0.5 1
}

launch_run_session() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }
  tmux new-session -d -s "$SESSION_NAME" -n retrieval \
    "cd '$REPO_ROOT' && bash '$0' --run >> '$LOG_ROOT/retrieval.log' 2>&1"
  echo "started session: $SESSION_NAME"
  echo "logs: $LOG_ROOT/retrieval.log"
}

waiter() {
  cd "$REPO_ROOT"
  local log="$LOG_ROOT/_waiter.log"
  exec > >(tee -a "$log") 2>&1
  echo "=== DEMO RETRIEVAL WAITER START $(date '+%F %T') ==="
  while active_blockers; do
    echo "[$(date '+%F %T')] blockers active; sleeping ${POLL_SEC}s"
    tmux ls 2>/dev/null || true
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits || true
    sleep "$POLL_SEC"
  done
  echo "[$(date '+%F %T')] blockers clear; launching demo retrieval probe"
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
  --run)
    run_sequence
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
