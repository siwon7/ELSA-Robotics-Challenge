#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${1:-fl_diffusion_lora4_jv_pilot}"

NUM_ROUNDS="${NUM_ROUNDS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
FRACTION_FIT="${FRACTION_FIT:-0.05}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-2.0}"
CLIENT_NUM_GPUS="${CLIENT_NUM_GPUS:-0.2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-24}"

CFG_FEDAVG="experiments/fl_dinov3_diffusion_lora4_jvdirect_fedavg.yaml"
CFG_FEDPROX="experiments/fl_dinov3_diffusion_lora4_jvdirect_fedprox.yaml"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME" -n slide_fedavg
tmux send-keys -t "$SESSION_NAME:slide_fedavg" \
  "cd '$REPO_ROOT' && ELSA_DATASET_CONFIG_PATH_OVERRIDE='$CFG_FEDAVG' CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' slide_block_to_target 0 '$NUM_ROUNDS' '$LOCAL_EPOCHS' 'dinov3-diffusion-lora4-jv-fedavg-r${NUM_ROUNDS}e${LOCAL_EPOCHS}-v1' '$FRACTION_FIT' '$TRAIN_SPLIT' '0.0'" C-m

tmux new-window -t "$SESSION_NAME" -n slide_fedprox
tmux send-keys -t "$SESSION_NAME:slide_fedprox" \
  "cd '$REPO_ROOT' && ELSA_DATASET_CONFIG_PATH_OVERRIDE='$CFG_FEDPROX' CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' slide_block_to_target 1 '$NUM_ROUNDS' '$LOCAL_EPOCHS' 'dinov3-diffusion-lora4-jv-fedprox-r${NUM_ROUNDS}e${LOCAL_EPOCHS}-v1' '$FRACTION_FIT' '$TRAIN_SPLIT' '0.001'" C-m

tmux new-window -t "$SESSION_NAME" -n close_fedavg
tmux send-keys -t "$SESSION_NAME:close_fedavg" \
  "cd '$REPO_ROOT' && ELSA_DATASET_CONFIG_PATH_OVERRIDE='$CFG_FEDAVG' CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' close_box 2 '$NUM_ROUNDS' '$LOCAL_EPOCHS' 'dinov3-diffusion-lora4-jv-fedavg-r${NUM_ROUNDS}e${LOCAL_EPOCHS}-v1' '$FRACTION_FIT' '$TRAIN_SPLIT' '0.0'" C-m

tmux new-window -t "$SESSION_NAME" -n close_fedprox
tmux send-keys -t "$SESSION_NAME:close_fedprox" \
  "cd '$REPO_ROOT' && ELSA_DATASET_CONFIG_PATH_OVERRIDE='$CFG_FEDPROX' CLIENT_NUM_CPUS='$CLIENT_NUM_CPUS' CLIENT_NUM_GPUS='$CLIENT_NUM_GPUS' RAY_NUM_CPUS='$RAY_NUM_CPUS' bash '$SCRIPT_DIR/run_flower_programmatic_one_task.sh' close_box 3 '$NUM_ROUNDS' '$LOCAL_EPOCHS' 'dinov3-diffusion-lora4-jv-fedprox-r${NUM_ROUNDS}e${LOCAL_EPOCHS}-v1' '$FRACTION_FIT' '$TRAIN_SPLIT' '0.001'" C-m

echo "Started tmux session: $SESSION_NAME"
