#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ELSA_TASK="${ELSA_TASK:-slide_block_to_target}"
export ELSA_STORAGE_ROOT="${ELSA_STORAGE_ROOT:-/mnt/ext_sdb1/elsa_robotics_challenge}"
export ELSA_DATA_ROOT="${ELSA_DATA_ROOT:-$ELSA_STORAGE_ROOT/datasets}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="${WANDB_DIR:-$ELSA_STORAGE_ROOT/wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
export FLWR_HOME="${FLWR_HOME:-$ELSA_STORAGE_ROOT/flwr_${ELSA_TASK}_3gpu_paperish}"
export TMPDIR="${TMPDIR:-/mnt/ext_sdb1/raytmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/cv7/.local/bin/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export ELSA_NUM_SUPERNODES="${ELSA_NUM_SUPERNODES:-400}"
export ELSA_CLIENT_NUM_CPUS="${ELSA_CLIENT_NUM_CPUS:-2}"
export ELSA_CLIENT_NUM_GPUS="${ELSA_CLIENT_NUM_GPUS:-0.188}"
export ELSA_NUM_SERVER_ROUNDS="${ELSA_NUM_SERVER_ROUNDS:-30}"
export ELSA_LOCAL_EPOCHS="${ELSA_LOCAL_EPOCHS:-50}"
export ELSA_FRACTION_FIT="${ELSA_FRACTION_FIT:-0.05}"
export ELSA_FRACTION_EVAL="${ELSA_FRACTION_EVAL:-0.0025}"
export ELSA_USE_WANDB="${ELSA_USE_WANDB:-true}"
export ELSA_POLICY_NAME="${ELSA_POLICY_NAME:-fk_camera_object}"
export ELSA_SERVER_DEVICE="${ELSA_SERVER_DEVICE:-cuda:0}"
export ELSA_SEED="${ELSA_SEED:-0}"
export ELSA_DETERMINISTIC_TRAINING="${ELSA_DETERMINISTIC_TRAINING:-false}"
export ELSA_EXTRA_RUN_CONFIG="${ELSA_EXTRA_RUN_CONFIG:-}"
FLOWER_CONFIG_PATH="$FLWR_HOME/config.toml"

mkdir -p \
  "$ELSA_DATA_ROOT" \
  "$ELSA_STORAGE_ROOT/model_checkpoints" \
  "$ELSA_STORAGE_ROOT/results" \
  "$WANDB_DIR" \
  "$FLWR_HOME" \
  "$TMPDIR" \
  "$RAY_TMPDIR"

cat > "$FLOWER_CONFIG_PATH" <<EOF
[superlink]
default = "local"

[superlink.supergrid]
address = "supergrid.flower.ai"

[superlink.local]
options.num-supernodes = ${ELSA_NUM_SUPERNODES}
options.backend.client-resources.num-cpus = ${ELSA_CLIENT_NUM_CPUS}
options.backend.client-resources.num-gpus = ${ELSA_CLIENT_NUM_GPUS}
EOF

source "$ROOT_DIR/.venv/bin/activate"
cd "$ROOT_DIR"
RUN_CONFIG="dataset-task=\"${ELSA_TASK}\" num-server-rounds=${ELSA_NUM_SERVER_ROUNDS} local-epochs=${ELSA_LOCAL_EPOCHS} fraction-fit=${ELSA_FRACTION_FIT} fraction-eval=${ELSA_FRACTION_EVAL} use-wandb=${ELSA_USE_WANDB} policy-name=\"${ELSA_POLICY_NAME}\" server-device=\"${ELSA_SERVER_DEVICE}\" seed=${ELSA_SEED} deterministic-training=${ELSA_DETERMINISTIC_TRAINING}"
if [ -n "$ELSA_EXTRA_RUN_CONFIG" ]; then
  RUN_CONFIG="${RUN_CONFIG} ${ELSA_EXTRA_RUN_CONFIG}"
fi
flwr run . --stream \
  --run-config "$RUN_CONFIG"
