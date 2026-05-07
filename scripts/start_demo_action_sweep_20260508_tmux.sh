#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/demo_action_sweep_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/demo_action_sweep_20260508"
SESSION_NAME="${DEMO_ACTION_SWEEP_SESSION:-demo_action_sweep_20260508}"
TARGET_SR="${DEMO_ACTION_SWEEP_TARGET_SR:-0.9}"
SCREEN_EPISODES="${DEMO_ACTION_SWEEP_SCREEN_EPISODES:-5}"
CONFIRM_EPISODES="${DEMO_ACTION_SWEEP_CONFIRM_EPISODES:-20}"
CONFIRM_TRIGGER_SR="${DEMO_ACTION_SWEEP_CONFIRM_TRIGGER_SR:-0.8}"
MAX_STEPS="${DEMO_ACTION_SWEEP_MAX_STEPS:-300}"
INDEX_SPLIT="${DEMO_ACTION_SWEEP_INDEX_SPLIT:-train}"
EVAL_SCRIPT="${DEMO_ACTION_SWEEP_EVAL_SCRIPT:-eval_demo_retrieval_policy_live.py}"
RUN_PREFIX="${DEMO_ACTION_SWEEP_RUN_PREFIX:-}"
EXTRA_EVAL_ARGS="${DEMO_ACTION_SWEEP_EXTRA_EVAL_ARGS:-}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"

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

sanitize_name() {
  echo "$1" | tr '/:. ' '____'
}

run_eval() {
  local slot="$1"
  local task="$2"
  local candidate="$3"
  local cfg="$4"
  local episodes="$5"
  local phase_weight="$6"
  local top_k="$7"
  local stage="$8"

  local run_name
  run_name="$(sanitize_name "${RUN_PREFIX}${candidate}_${stage}_e${episodes}_pw${phase_weight}_k${top_k}_${INDEX_SPLIT}")"
  local output="$RESULT_ROOT/$task/$run_name/result.json"
  if [ -s "$output" ]; then
    echo "skip existing: task=$task run=$run_name"
    sr_from_result "$output"
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

  local extra_args=()
  if [ -n "$EXTRA_EVAL_ARGS" ]; then
    # shellcheck disable=SC2206
    extra_args=($EXTRA_EVAL_ARGS)
  fi

  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/$EVAL_SCRIPT"
    --task "$task"
    --dataset-config-path "$REPO_ROOT/$cfg"
    --rollout-split training
    --env-ids 0
    --index-env-ids 0
    --index-split "$INDEX_SPLIT"
    --episodes "$episodes"
    --max-steps "$MAX_STEPS"
    --state-dim 8
    --phase-weight "$phase_weight"
    --top-k "$top_k"
    "${extra_args[@]}"
    --output "$output"
  )

  echo "=== ACTION SWEEP START slot=$slot task=$task candidate=$candidate stage=$stage script=$EVAL_SCRIPT cfg=$cfg episodes=$episodes phase=$phase_weight top_k=$top_k extra='$EXTRA_EVAL_ARGS' $(date '+%F %T') ==="
  export CUDA_VISIBLE_DEVICES=""
  export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
  export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
  elsa_run_with_cpu_limit "$slot" "${cmd[@]}"
  echo "=== ACTION SWEEP END slot=$slot task=$task candidate=$candidate stage=$stage $(date '+%F %T') ==="
  sr_from_result "$output"
}

append_summary() {
  local task="$1"
  local candidate="$2"
  local cfg="$3"
  local stage="$4"
  local episodes="$5"
  local phase_weight="$6"
  local top_k="$7"
  local sr="$8"
  local summary="$RESULT_ROOT/_summary.tsv"
  if [ ! -s "$summary" ]; then
    echo -e "time\ttask\tcandidate\tstage\tepisodes\tphase_weight\ttop_k\tsr\tconfig" >> "$summary"
  fi
  echo -e "$(date '+%F %T')\t$task\t$candidate\t$stage\t$episodes\t$phase_weight\t$top_k\t$sr\t$cfg" >> "$summary"
}

run_candidate() {
  local slot="$1"
  local task="$2"
  local candidate="$3"
  local cfg="$4"
  local phase_weight="$5"
  local top_k="$6"

  local screen_sr
  screen_sr="$(run_eval "$slot" "$task" "$candidate" "$cfg" "$SCREEN_EPISODES" "$phase_weight" "$top_k" screen | tail -1)"
  append_summary "$task" "$candidate" "$cfg" screen "$SCREEN_EPISODES" "$phase_weight" "$top_k" "$screen_sr"

  if [ "$(sr_ge "$screen_sr" "$CONFIRM_TRIGGER_SR")" != "1" ]; then
    return 1
  fi

  local confirm_sr
  confirm_sr="$(run_eval "$slot" "$task" "$candidate" "$cfg" "$CONFIRM_EPISODES" "$phase_weight" "$top_k" confirm | tail -1)"
  append_summary "$task" "$candidate" "$cfg" confirm "$CONFIRM_EPISODES" "$phase_weight" "$top_k" "$confirm_sr"

  if [ "$(sr_ge "$confirm_sr" "$TARGET_SR")" = "1" ]; then
    echo "$candidate	$confirm_sr	$cfg	phase=$phase_weight	top_k=$top_k" > "$RESULT_ROOT/$task/BEST_ACTION.txt"
    return 0
  fi
  return 1
}

run_task_worker() {
  local slot="$1"
  local task="$2"
  shift 2

  cd "$REPO_ROOT"
  mkdir -p "$RESULT_ROOT/$task"
  echo "=== ACTION SWEEP WORKER START slot=$slot task=$task target_sr=$TARGET_SR index_split=$INDEX_SPLIT $(date '+%F %T') ==="
  local best_file="$RESULT_ROOT/$task/BEST_ACTION.txt"
  if [ -s "$best_file" ] && ! grep -q "^NO_ACTION_HIT" "$best_file"; then
    local best_sr
    best_sr="$(awk 'NR==1 {print $2}' "$best_file")"
    if [ "$(sr_ge "$best_sr" "$TARGET_SR")" = "1" ]; then
      echo "skip task=$task existing best_sr=$best_sr target_sr=$TARGET_SR"
      return 0
    fi
  fi

  local candidate cfg phase_weight top_k
  for spec in "$@"; do
    IFS='|' read -r candidate cfg phase_weight top_k <<< "$spec"
    if run_candidate "$slot" "$task" "$candidate" "$cfg" "$phase_weight" "$top_k"; then
      echo "=== ACTION SWEEP HIT task=$task candidate=$candidate target_sr=$TARGET_SR $(date '+%F %T') ==="
      return 0
    fi
  done

  echo "NO_ACTION_HIT target_sr=$TARGET_SR" > "$RESULT_ROOT/$task/BEST_ACTION.txt"
  echo "=== ACTION SWEEP EXHAUSTED task=$task target_sr=$TARGET_SR $(date '+%F %T') ==="
  return 0
}

common_candidates=(
  "jv_direct|experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml|0.5|1"
  "jv_direct_phase1|experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml|1.0|1"
  "jv_w4_exec2|experiments/sameenv_dino_depth_diffusion_lora8_jvdirect_chunk4exec2.yaml|0.5|1"
  "jv_w4_exec2_top3|experiments/sameenv_dino_depth_diffusion_lora8_jvdirect_chunk4exec2.yaml|0.5|3"
  "jprel_w2_direct|experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w2_direct_grid16_eeaux.yaml|0.5|1"
  "jprel_w4_direct|experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux.yaml|0.5|1"
  "jprel_w4_direct_top3|experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux.yaml|0.5|3"
  "jprel_w4_jvservo|experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_jvservo_grid16_eeaux.yaml|0.5|1"
  "jp_keyframe4_servo|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpkeyframe4_jpservo.yaml|0.5|1"
  "jp_keyframe4_servo_phase1|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpkeyframe4_jpservo.yaml|1.0|1"
  "jp_keyframe4_servo_top3|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpkeyframe4_jpservo.yaml|0.5|3"
)

task_candidates() {
  local task="$1"
  case "$task" in
    slide_block_to_target)
      printf '%s\n' \
        "jp_servo|experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo.yaml|0.5|1" \
        "jp_servo_w4_exec2|experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|0.5|1" \
        "jp_direct|experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml|0.5|1" \
        "jp_direct_w4_exec2|experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2.yaml|0.5|1" \
        "jp_keyframe4_direct|experiments/slide_block_to_target_sameenv_action_keyframe4_dinov3_jpdirect.yaml|0.5|1" \
        "${common_candidates[@]}"
      ;;
    close_box)
      printf '%s\n' \
        "jp_servo|experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo.yaml|0.5|1" \
        "jp_servo_w4_exec2|experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|0.5|1" \
        "jp_direct|experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml|0.5|1" \
        "jp_direct_w4_exec2|experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2.yaml|0.5|1" \
        "${common_candidates[@]}"
      ;;
    insert_onto_square_peg)
      printf '%s\n' \
        "jp_servo|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpservo.yaml|0.5|1" \
        "jp_servo_w4_exec2|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|0.5|1" \
        "jp_direct|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml|0.5|1" \
        "jp_direct_w4_exec2|experiments/insert_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2.yaml|0.5|1" \
        "${common_candidates[@]}"
      ;;
    scoop_with_spatula)
      printf '%s\n' \
        "jp_servo|experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpservo.yaml|0.5|1" \
        "jp_servo_w4_exec2|experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml|0.5|1" \
        "jp_direct|experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml|0.5|1" \
        "jp_direct_w4_exec2|experiments/scoop_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2.yaml|0.5|1" \
        "${common_candidates[@]}"
      ;;
    *)
      echo "unsupported task: $task" >&2
      return 2
      ;;
  esac
}

worker() {
  local slot="$1"
  local task="$2"
  mapfile -t specs < <(task_candidates "$task")
  run_task_worker "$slot" "$task" "${specs[@]}"
}

launch_run_session() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }

  worker_cmd() {
    local slot="$1"
    local task="$2"
    local log="$3"
    printf "cd %q && env DEMO_ACTION_SWEEP_INDEX_SPLIT=%q DEMO_ACTION_SWEEP_EVAL_SCRIPT=%q DEMO_ACTION_SWEEP_RUN_PREFIX=%q DEMO_ACTION_SWEEP_EXTRA_EVAL_ARGS=%q DEMO_ACTION_SWEEP_SCREEN_EPISODES=%q DEMO_ACTION_SWEEP_CONFIRM_EPISODES=%q DEMO_ACTION_SWEEP_CONFIRM_TRIGGER_SR=%q DEMO_ACTION_SWEEP_TARGET_SR=%q bash %q --worker %q %q >> %q 2>&1" \
      "$REPO_ROOT" \
      "$INDEX_SPLIT" \
      "$EVAL_SCRIPT" \
      "$RUN_PREFIX" \
      "$EXTRA_EVAL_ARGS" \
      "$SCREEN_EPISODES" \
      "$CONFIRM_EPISODES" \
      "$CONFIRM_TRIGGER_SR" \
      "$TARGET_SR" \
      "$0" \
      "$slot" \
      "$task" \
      "$log"
  }

  tmux new-session -d -s "$SESSION_NAME" -n slide \
    "$(worker_cmd 0 slide_block_to_target "$LOG_ROOT/slide.log")"
  tmux new-window -t "$SESSION_NAME" -n close \
    "$(worker_cmd 1 close_box "$LOG_ROOT/close.log")"
  tmux new-window -t "$SESSION_NAME" -n insert \
    "$(worker_cmd 2 insert_onto_square_peg "$LOG_ROOT/insert.log")"
  tmux new-window -t "$SESSION_NAME" -n scoop \
    "$(worker_cmd 3 scoop_with_spatula "$LOG_ROOT/scoop.log")"

  echo "started session: $SESSION_NAME"
  echo "logs: $LOG_ROOT"
  tmux list-windows -t "$SESSION_NAME"
}

case "${1:-}" in
  --worker)
    worker "${2:?usage: $0 --worker <slot> <task>}" "${3:?usage: $0 --worker <slot> <task>}"
    ;;
  *)
    launch_run_session
    ;;
esac
