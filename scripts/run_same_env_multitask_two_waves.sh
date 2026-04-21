#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <config1> <run_prefix1> <config2> <run_prefix2>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG1="$1"
PREFIX1="$2"
CONFIG2="$3"
PREFIX2="$4"

EPOCHS="${EPOCHS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
SEED="${SEED:-0}"
RESULT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}/results/same_env_suite"

wait_for_wave() {
  local prefix="$1"
  local tasks=("close_box" "insert_onto_square_peg" "scoop_with_spatula" "slide_block_to_target")
  while true; do
    local done=1
    for task in "${tasks[@]}"; do
      local path="$RESULT_ROOT/$task/${task}_${prefix}_e${EPOCHS}_s${SEED}/env_000/result.json"
      if [ ! -f "$path" ]; then
        done=0
        break
      fi
    done
    if [ "$done" -eq 1 ]; then
      break
    fi
    sleep 60
  done
}

cd "$REPO_ROOT"
EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  bash "$SCRIPT_DIR/start_same_env_multitask_config_tmux.sh" "$CONFIG1" "$PREFIX1" "same_env_multitask_${PREFIX1}"
wait_for_wave "$PREFIX1"

EPOCHS="$EPOCHS" EVAL_EPISODES="$EVAL_EPISODES" SEED="$SEED" \
  bash "$SCRIPT_DIR/start_same_env_multitask_config_tmux.sh" "$CONFIG2" "$PREFIX2" "same_env_multitask_${PREFIX2}"
wait_for_wave "$PREFIX2"

python "$SCRIPT_DIR/collect_same_env_multitask_results.py" --run-prefix "$PREFIX1" --epochs "$EPOCHS" --seed "$SEED"
python "$SCRIPT_DIR/collect_same_env_multitask_results.py" --run-prefix "$PREFIX2" --epochs "$EPOCHS" --seed "$SEED"
