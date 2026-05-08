#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/action_replay_ceiling_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/action_replay_ceiling_20260508"
SESSION_NAME="${ACTION_REPLAY_CEILING_SESSION:-action_replay_ceiling_20260508}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
NUM_DEMOS="${ACTION_REPLAY_CEILING_NUM_DEMOS:-3}"
TARGET_SR="${ACTION_REPLAY_CEILING_TARGET_SR:-0.9}"
MAX_PACK_TIME_SEC="${ACTION_REPLAY_CEILING_MAX_PACK_TIME_SEC:-120}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

sr_from_result() {
  local output="$1"
  python - "$output" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("-1")
    raise SystemExit
with path.open("r", encoding="utf-8") as fh:
    payload = json.load(fh)
print(float(payload.get("sr", -1)))
PY
}

sr_ge() {
  python - "$1" "$2" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)
PY
}

append_summary() {
  local task="$1"
  local probe="$2"
  local sr="$3"
  local output="$4"
  local summary="$RESULT_ROOT/_summary.tsv"
  if [ ! -s "$summary" ]; then
    echo -e "time\ttask\tprobe\tsr\toutput" >> "$summary"
  fi
  echo -e "$(date '+%F %T')\t$task\t$probe\t$sr\t$output" >> "$summary"
}

prepare_env() {
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/cpu_limit_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"
  export ELSA_PYTHON_BIN="/home/cvlab-dgx/anaconda3/envs/${ENV_NAME}/bin/python"
  if [ ! -x "$ELSA_PYTHON_BIN" ]; then
    ELSA_PYTHON_BIN="$(command -v python)"
    export ELSA_PYTHON_BIN
  fi
  export CUDA_VISIBLE_DEVICES=""
  export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
  export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
}

run_collect_packs() {
  local task="$1"
  local task_root="$RESULT_ROOT/$task"
  local output="$task_root/live_packs/result.json"
  local pack_count
  mkdir -p "$task_root/live_packs"
  pack_count="$(find "$task_root/live_packs" -maxdepth 1 -name '*.replay.pkl' 2>/dev/null | wc -l)"
  if [ -s "$output" ] && [ "$pack_count" -ge "$NUM_DEMOS" ]; then
    echo "skip collect existing task=$task packs=$pack_count"
    append_summary "$task" "collect_live_packs" "$(sr_from_result "$output")" "$output"
    return 0
  fi

  echo "=== COLLECT LIVE PACKS START task=$task demos=$NUM_DEMOS $(date '+%F %T') ==="
  prepare_env
  elsa_run_with_cpu_limit 3 \
    "$ELSA_PYTHON_BIN" "$REPO_ROOT/scripts/eval_live_expert_reproduction.py" \
      --task "$task" \
      --split training \
      --env-ids 0 \
      --num-demos "$NUM_DEMOS" \
      --method expert_success \
      --output "$output"
  local sr
  sr="$(sr_from_result "$output")"
  append_summary "$task" "collect_live_packs" "$sr" "$output"
  echo "=== COLLECT LIVE PACKS END task=$task sr=$sr $(date '+%F %T') ==="
}

run_replay_probe() {
  local task="$1"
  local arm_mode="$2"
  local hold_steps="$3"
  local task_root="$RESULT_ROOT/$task"
  local output="$task_root/replay_${arm_mode}_hold${hold_steps}/result.json"
  if [ -s "$output" ]; then
    local existing_sr
    existing_sr="$(sr_from_result "$output")"
    echo "skip replay existing task=$task arm_mode=$arm_mode hold=$hold_steps sr=$existing_sr"
    append_summary "$task" "replay_${arm_mode}_hold${hold_steps}" "$existing_sr" "$output"
    return 0
  fi

  echo "=== REPLAY PROBE START task=$task arm_mode=$arm_mode hold=$hold_steps $(date '+%F %T') ==="
  prepare_env
  mkdir -p "$(dirname "$output")"
  elsa_run_with_cpu_limit 3 \
    "$ELSA_PYTHON_BIN" "$REPO_ROOT/scripts/eval_saved_replay_unified.py" \
      --task "$task" \
      --split training \
      --pack-dir "$task_root/live_packs" \
      --arm-mode "$arm_mode" \
      --hold-steps "$hold_steps" \
      --max-pack-time-sec "$MAX_PACK_TIME_SEC" \
      --output "$output"
  local sr
  sr="$(sr_from_result "$output")"
  append_summary "$task" "replay_${arm_mode}_hold${hold_steps}" "$sr" "$output"
  if [ "$(sr_ge "$sr" "$TARGET_SR")" = "1" ]; then
    echo "$arm_mode hold=$hold_steps sr=$sr output=$output" >> "$task_root/BEST_REPLAY_ACTION.txt"
  fi
  echo "=== REPLAY PROBE END task=$task arm_mode=$arm_mode hold=$hold_steps sr=$sr $(date '+%F %T') ==="
}

worker() {
  cd "$REPO_ROOT"
  exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
  echo "=== ACTION REPLAY CEILING START target_sr=$TARGET_SR demos=$NUM_DEMOS $(date '+%F %T') ==="
  local tasks=(
    slide_block_to_target
    close_box
    insert_onto_square_peg
    scoop_with_spatula
  )
  local modes=(
    jv
    jv_finite_diff
    jp_abs
    jp_rel
  )
  local holds=(
    1
    2
  )
  for task in "${tasks[@]}"; do
    run_collect_packs "$task"
    for arm_mode in "${modes[@]}"; do
      for hold_steps in "${holds[@]}"; do
        run_replay_probe "$task" "$arm_mode" "$hold_steps"
      done
    done
  done
  echo "=== ACTION REPLAY CEILING DONE $(date '+%F %T') ==="
  find "$RESULT_ROOT" -maxdepth 2 -name BEST_REPLAY_ACTION.txt -print -exec sed -n '1,20p' {} \;
}

launch() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }
  tmux new-session -d -s "$SESSION_NAME" -n driver \
    "cd '$REPO_ROOT' && bash '$0' --worker"
  echo "started session: $SESSION_NAME"
  echo "logs: $LOG_ROOT/driver.log"
}

case "${1:-}" in
  --worker)
    worker
    ;;
  *)
    launch
    ;;
esac
