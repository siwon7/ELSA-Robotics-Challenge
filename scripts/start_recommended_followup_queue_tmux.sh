#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/recommended_followups_20260504"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/recommended_followups_20260504"
LOG_ROOT="$ARTIFACT_ROOT/logs/recommended_followups_20260504"
WAIT_SESSION="${RECOMMENDED_WAIT_SESSION:-recommended_followups_wait}"
RUN_SESSION="${RECOMMENDED_RUN_SESSION:-recommended_followups_20260504}"
POLL_SEC="${POLL_SEC:-300}"

TRAIN_ENV_IDS=(0 1 2 3 4)
EVAL_ENV_IDS=(0 1 2 3 4)

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

blockers_running() {
  pgrep -f "scripts/start_overnight_queue_pending_tmux.sh --worker" >/dev/null 2>&1 && return 0
  pgrep -f "scripts/wait_and_run_volumedp_paperclose.sh" >/dev/null 2>&1 && return 0
  return 1
}

result_exists() {
  local task="$1"
  local run_name="$2"
  compgen -G "$RESULT_ROOT/$task/$run_name/*/result.json" >/dev/null
}

sr_from_glob() {
  local pattern="$1"
  python - "$pattern" <<'PY'
import glob
import json
import sys

matches = glob.glob(sys.argv[1])
if not matches:
    print("nan")
    raise SystemExit

with open(matches[0], "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get("sr", data.get("mean_per_env_sr"))
print("nan" if value is None else value)
PY
}

sr_ge() {
  local value="$1"
  local threshold="$2"
  python - "$value" "$threshold" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
    threshold = float(sys.argv[2])
except ValueError:
    raise SystemExit(1)

raise SystemExit(0 if math.isfinite(value) and value >= threshold else 1)
PY
}

run_train() {
  local gpu="$1"
  local task="$2"
  local cfg="$3"
  local epochs="$4"
  local seed="$5"
  local run_name="$6"
  local kind="$7"

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

  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py"
    --task "$task"
    --dataset-config-path "$REPO_ROOT/$cfg"
    --epochs "$epochs"
    --eval-episodes 20
    --device cuda:0
    --seed "$seed"
    --run-name "$run_name"
    --output-root "$RESULT_ROOT"
    --checkpoint-root "$CKPT_ROOT"
  )

  if [ "$kind" = "multi5" ]; then
    cmd+=(--train-env-ids "${TRAIN_ENV_IDS[@]}")
    cmd+=(--eval-env-ids "${EVAL_ENV_IDS[@]}")
  fi

  echo "=== START $run_name gpu=$gpu task=$task kind=$kind e$epochs seed=$seed $(date '+%F %T') ==="
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/${run_name}.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$run_name exit=$status" | tee -a "$LOG_ROOT/_recommended_status_gpu${gpu}.log"
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
        experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml \
        100 0 slide_volumedp_w4_eeaux_e100_s0 same
      run_train 0 slide_block_to_target \
        experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml \
        10 0 slide_volumedp_w4_eeaux_5env_e10_s0 multi5
      ;;
    1)
      run_train 1 slide_block_to_target \
        experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml \
        50 1 slide_volumedp_w4_eeaux_e50_s1 same
      ;;
    2)
      run_train 2 slide_block_to_target \
        experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml \
        50 2 slide_volumedp_w4_eeaux_e50_s2 same
      ;;
    3)
      local close_sr
      close_sr="$(sr_from_glob "$ARTIFACT_ROOT/results/overnight_queue/close_box/close_volumedp_w4_eeaux_e50/*/result.json")"
      echo "close_volumedp_w4_eeaux_e50 sr=$close_sr"
      if sr_ge "$close_sr" 0.15; then
        run_train 3 close_box \
          experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_w4_tight_grid16_eeaux.yaml \
          100 0 close_volumedp_w4_eeaux_e100_s0 same
      else
        echo "skip close_volumedp_w4_eeaux_e100_s0 because e50 sr=$close_sr < 0.15"
      fi
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
  tmux new-window -t "$RUN_SESSION" -n gpu1_slide_seed1
  tmux new-window -t "$RUN_SESSION" -n gpu2_slide_seed2
  tmux new-window -t "$RUN_SESSION" -n gpu3_close_cond

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

  echo "=== recommended followup waiter start $(date '+%F %T') ==="
  while blockers_running; do
    echo "[$(date '+%F %T')] existing overnight/paperclose queue still running; sleeping ${POLL_SEC}s"
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
