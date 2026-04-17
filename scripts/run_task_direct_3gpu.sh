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
export ELSA_DATASET_CONFIG_PATH="${ELSA_DATASET_CONFIG_PATH:-dataset_config.yaml}"
export ELSA_TRAIN_SPLIT="${ELSA_TRAIN_SPLIT:-0.9}"
export ELSA_SAVE_ROUNDS="${ELSA_SAVE_ROUNDS:-5,25,50,100}"
export ELSA_STRATEGY_NAME="${ELSA_STRATEGY_NAME:-fedavg}"
export ELSA_VERBOSE_SIM="${ELSA_VERBOSE_SIM:-false}"
export ELSA_ENABLE_CENTRALIZED_EVAL="${ELSA_ENABLE_CENTRALIZED_EVAL:-false}"
export ELSA_CENTRALIZED_EVAL_SIMULATOR="${ELSA_CENTRALIZED_EVAL_SIMULATOR:-false}"
export ELSA_CENTRALIZED_EVAL_BATCH_SIZE="${ELSA_CENTRALIZED_EVAL_BATCH_SIZE:-32}"
export ELSA_CENTRALIZED_EVAL_NUM_WORKERS="${ELSA_CENTRALIZED_EVAL_NUM_WORKERS:-8}"
export ELSA_ENABLE_CLIENT_LOCAL_STATE="${ELSA_ENABLE_CLIENT_LOCAL_STATE:-true}"

mkdir -p \
  "$ELSA_DATA_ROOT" \
  "$ELSA_STORAGE_ROOT/model_checkpoints" \
  "$ELSA_STORAGE_ROOT/results" \
  "$WANDB_DIR" \
  "$FLWR_HOME" \
  "$TMPDIR" \
  "$RAY_TMPDIR"

source "$ROOT_DIR/.venv/bin/activate"
cd "$ROOT_DIR"

BOOL_ARGS=()
if [ "$ELSA_USE_WANDB" = "true" ]; then
  BOOL_ARGS+=(--use-wandb)
else
  BOOL_ARGS+=(--no-use-wandb)
fi

if [ "$ELSA_DETERMINISTIC_TRAINING" = "true" ]; then
  BOOL_ARGS+=(--deterministic-training)
else
  BOOL_ARGS+=(--no-deterministic-training)
fi

if [ "$ELSA_VERBOSE_SIM" = "true" ]; then
  BOOL_ARGS+=(--verbose)
else
  BOOL_ARGS+=(--no-verbose)
fi

python scripts/run_flwr_simulation.py \
  --app-dir "$ROOT_DIR" \
  --dataset-config-path "$ELSA_DATASET_CONFIG_PATH" \
  --dataset-task "$ELSA_TASK" \
  --policy-name "$ELSA_POLICY_NAME" \
  --server-device "$ELSA_SERVER_DEVICE" \
  --strategy-name "$ELSA_STRATEGY_NAME" \
  --num-supernodes "$ELSA_NUM_SUPERNODES" \
  --num-server-rounds "$ELSA_NUM_SERVER_ROUNDS" \
  --local-epochs "$ELSA_LOCAL_EPOCHS" \
  --fraction-fit "$ELSA_FRACTION_FIT" \
  --fraction-eval "$ELSA_FRACTION_EVAL" \
  --train-split "$ELSA_TRAIN_SPLIT" \
  --save-rounds "$ELSA_SAVE_ROUNDS" \
  --client-num-cpus "$ELSA_CLIENT_NUM_CPUS" \
  --client-num-gpus "$ELSA_CLIENT_NUM_GPUS" \
  --seed "$ELSA_SEED" \
  --ray-temp-dir "$RAY_TMPDIR" \
  "${BOOL_ARGS[@]}"
