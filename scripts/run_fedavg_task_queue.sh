#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <gpu_id> <local_epochs> <num_server_rounds> <log_dir> <control_port_base> <simulation_port_base>"
  exit 1
fi

GPU_ID="$1"
LOCAL_EPOCHS="$2"
NUM_SERVER_ROUNDS="$3"
LOG_DIR="$4"
CONTROL_BASE="$5"
SIM_BASE="$6"

ROOT="/home/cv25/siwon/ELSA-Robotics-Challenge"
mkdir -p "$LOG_DIR"

TASKS=(
  "close_box"
  "slide_block_to_target"
  "insert_onto_square_peg"
  "scoop_with_spatula"
)

cd "$ROOT"

for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  log_file="$LOG_DIR/${task}.log"
  control_port=$((CONTROL_BASE + idx * 10))
  simulation_port=$((SIM_BASE + idx * 10))
  echo "[$(date '+%F %T')] starting task=${task} gpu=${GPU_ID} local_epochs=${LOCAL_EPOCHS} rounds=${NUM_SERVER_ROUNDS}" | tee -a "$log_file"
  bash "$ROOT/scripts/run_strategy_one_task.sh" \
    "$GPU_ID" \
    "$task" \
    "fedavg" \
    "$log_file" \
    "$control_port" \
    "$simulation_port" \
    "$LOCAL_EPOCHS" \
    "$NUM_SERVER_ROUNDS"
  echo "[$(date '+%F %T')] finished task=${task}" | tee -a "$log_file"
done
