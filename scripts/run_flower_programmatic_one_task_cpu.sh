#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

ELSA_PYTHON_BIN="${ELSA_PYTHON_BIN:-$CONDA_BASE/envs/$ELSA_ENV_NAME/bin/python}"
if [[ ! -x "$ELSA_PYTHON_BIN" ]]; then
  echo "Expected python not found: $ELSA_PYTHON_BIN" >&2
  exit 1
fi

TASK_NAME="${1:?task name required}"
NUM_ROUNDS="${2:-2}"
LOCAL_EPOCHS="${3:-2}"
RUN_TAG="${4:-cpu-smoke-v1}"
FRACTION_FIT="${5:-0.02}"
TRAIN_SPLIT="${6:-0.9}"
PROX_MU="${7:-0.001}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-2.0}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
NUM_CLIENTS="${NUM_CLIENTS:-400}"
SUMMARY_ROOT="${SUMMARY_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/flower_cpu}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints_cpu}"

mkdir -p "$SUMMARY_ROOT" "$CHECKPOINT_ROOT"

export ELSA_DATASET_CONFIG_PATH="${ELSA_DATASET_CONFIG_PATH_OVERRIDE:-$REPO_ROOT/dataset_config.yaml}"
export ELSA_DATASET_TASK="$TASK_NAME"
export ELSA_TRAIN_SPLIT="$TRAIN_SPLIT"
export ELSA_LOCAL_EPOCHS="$LOCAL_EPOCHS"
export ELSA_CLIENT_DEVICE="cpu"
export ELSA_PROX_MU="$PROX_MU"

"$ELSA_PYTHON_BIN" "$SCRIPT_DIR/run_flower_programmatic_one_task.py" \
  --task "$TASK_NAME" \
  --rounds "$NUM_ROUNDS" \
  --local-epochs "$LOCAL_EPOCHS" \
  --fraction-fit "$FRACTION_FIT" \
  --fraction-eval 0.0 \
  --train-split "$TRAIN_SPLIT" \
  --prox-mu "$PROX_MU" \
  --dataset-config-path "$ELSA_DATASET_CONFIG_PATH" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --run-tag "$RUN_TAG" \
  --server-device "cpu" \
  --client-device "cpu" \
  --num-clients "$NUM_CLIENTS" \
  --client-num-cpus "$CLIENT_NUM_CPUS" \
  --client-num-gpus 0.0 \
  --ray-num-cpus "$RAY_NUM_CPUS" \
  --ray-num-gpus 0 \
  --summary-path "$SUMMARY_ROOT/${TASK_NAME}_${RUN_TAG}.json"
