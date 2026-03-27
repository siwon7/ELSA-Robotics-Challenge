#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cv25/siwon/ELSA-Robotics-Challenge"
LOG_DIR="$ROOT/logs/fedavg_tasks"
mkdir -p "$LOG_DIR"
STRATEGY="fedavg"

TASKS=(
  "close_box"
  "slide_block_to_target"
  "insert_onto_square_peg"
  "scoop_with_spatula"
)

CONTROL_PORTS=(39103 39113 39123 39133)
SIMULATION_PORTS=(39104 39114 39124 39134)

for gpu in 0 1 2 3; do
  task="${TASKS[$gpu]}"
  session="${STRATEGY}_${task}"
  log_file="$LOG_DIR/${session}.log"
  control_port="${CONTROL_PORTS[$gpu]}"
  simulation_port="${SIMULATION_PORTS[$gpu]}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Skipping existing tmux session: $session"
    continue
  fi

  tmux new-session -d -s "$session" \
    "bash /home/cv25/siwon/ELSA-Robotics-Challenge/scripts/run_strategy_one_task.sh $gpu $task $STRATEGY $log_file $control_port $simulation_port"
  echo "Started $session on GPU $gpu"
done

echo "Logs: $LOG_DIR"
