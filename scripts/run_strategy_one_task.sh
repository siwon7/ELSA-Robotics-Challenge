#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ] || [ "$#" -gt 8 ]; then
  echo "Usage: $0 <gpu_id> <task> <strategy> <log_file> <control_port> <simulationio_port> [local_epochs] [num_server_rounds]"
  exit 1
fi

GPU_ID="$1"
TASK="$2"
STRATEGY="$3"
LOG_FILE="$4"
CONTROL_PORT="$5"
SIMULATIONIO_PORT="$6"
LOCAL_EPOCHS="${7:-50}"
NUM_SERVER_ROUNDS="${8:-100}"

ROOT="/home/cv25/siwon/ELSA-Robotics-Challenge"
ENV_NAME="elsa-robotics-challenge"
FLWR_HOME_DIR="$ROOT/.flwr/${TASK}_${STRATEGY}_le${LOCAL_EPOCHS}_r${NUM_SERVER_ROUNDS}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$FLWR_HOME_DIR"
mkdir -p "$FLWR_HOME_DIR/local-superlink"

cat > "$FLWR_HOME_DIR/config.toml" <<EOF
[superlink]
default = "local"

[superlink.local]
options.num-supernodes = 400
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.05
options.backend.init-args.num-cpus = 8
options.backend.init-args.num-gpus = 1
EOF

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export FLWR_HOME="$FLWR_HOME_DIR"
export FLWR_LOCAL_CONTROL_API_PORT="$CONTROL_PORT"
export FLWR_LOCAL_SIMULATIONIO_API_PORT="$SIMULATIONIO_PORT"

conda run -n "$ENV_NAME" \
  flwr run . --stream \
  -c "dataset-task=\"$TASK\" strategy-name=\"$STRATEGY\" use-wandb=false server-device=\"cuda:0\" local-epochs=${LOCAL_EPOCHS} num-server-rounds=${NUM_SERVER_ROUNDS}" \
  2>&1 | tee "$LOG_FILE"
