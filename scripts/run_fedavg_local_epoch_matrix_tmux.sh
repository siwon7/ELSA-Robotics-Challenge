#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cv25/siwon/ELSA-Robotics-Challenge"
LOG_ROOT="$ROOT/logs/local_epoch_matrix"
mkdir -p "$LOG_ROOT"

LOCAL_EPOCHS=(5 25 50 100)
GPUS=(0 1 2 3)
CONTROL_BASES=(39203 39303 39403 39503)
SIM_BASES=(39204 39304 39404 39504)
NUM_SERVER_ROUNDS=100

for idx in "${!LOCAL_EPOCHS[@]}"; do
  le="${LOCAL_EPOCHS[$idx]}"
  gpu="${GPUS[$idx]}"
  control_base="${CONTROL_BASES[$idx]}"
  sim_base="${SIM_BASES[$idx]}"
  session="fedavg_le${le}_gpu${gpu}"
  log_dir="$LOG_ROOT/le${le}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Skipping existing tmux session: $session"
    continue
  fi

  tmux new-session -d -s "$session" \
    "bash $ROOT/scripts/run_fedavg_task_queue.sh $gpu $le $NUM_SERVER_ROUNDS $log_dir $control_base $sim_base"
  echo "Started $session on GPU $gpu"
done

echo "Logs: $LOG_ROOT"
