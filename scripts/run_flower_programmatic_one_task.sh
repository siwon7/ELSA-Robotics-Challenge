#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/ensure_local_xvfb.sh"

ELSA_PYTHON_BIN="${ELSA_PYTHON_BIN:-$CONDA_BASE/envs/$ELSA_ENV_NAME/bin/python}"
if [[ ! -x "$ELSA_PYTHON_BIN" ]]; then
  echo "Expected python not found: $ELSA_PYTHON_BIN" >&2
  exit 1
fi

TASK_NAME="${1:?task name required}"
GPU_INDEX="${2:?gpu index required}"
NUM_ROUNDS="${3:-3}"
LOCAL_EPOCHS="${4:-5}"
RUN_TAG="${5:-abs-jpos-prog-v1}"
FRACTION_FIT="${6:-0.05}"
TRAIN_SPLIT="${7:-0.9}"
PROX_MU="${8:-0.0}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-2.0}"
CLIENT_NUM_GPUS="${CLIENT_NUM_GPUS:-0.1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-24}"
ELSA_ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-60}"
LOG_ROOT="${ELSA_ARTIFACT_ROOT}/logs/flower_abs_joint_pos_programmatic"
CHECKPOINT_ROOT="${ELSA_ARTIFACT_ROOT}/model_checkpoints"
RESULTS_ROOT="${ELSA_ARTIFACT_ROOT}/results/flower_abs_joint_pos_programmatic"
TMP_ROOT="${TMP_ROOT:-/mnt/raid0/elsa_tmp}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/mnt/raid0/raytmp}"

mkdir -p "$LOG_ROOT"
mkdir -p "$CHECKPOINT_ROOT"
mkdir -p "$RESULTS_ROOT"
mkdir -p "$TMP_ROOT"
mkdir -p "$RAY_TMP_ROOT"
RUN_LOG="${LOG_ROOT}/${TASK_NAME}_${RUN_TAG}.log"

while true; do
  echo "[$(date --iso-8601=seconds)] launching isolated task=${TASK_NAME} gpu=${GPU_INDEX}" | tee -a "$RUN_LOG"
  set +e
  export ELSA_DATASET_CONFIG_PATH="${ELSA_DATASET_CONFIG_PATH_OVERRIDE:-$REPO_ROOT/dataset_config.yaml}"
  export ELSA_DATASET_TASK="$TASK_NAME"
  export ELSA_TRAIN_SPLIT="$TRAIN_SPLIT"
  export ELSA_LOCAL_EPOCHS="$LOCAL_EPOCHS"
  export ELSA_CLIENT_DEVICE="cuda:0"
  export ELSA_PROX_MU="$PROX_MU"
  export TMPDIR="${TMP_ROOT}"
  export TMP="${TMP_ROOT}"
  export TEMP="${TMP_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  "$ELSA_PYTHON_BIN" "$SCRIPT_DIR/run_flower_programmatic_one_task.py" \
    --task "$TASK_NAME" \
    --rounds "$NUM_ROUNDS" \
    --local-epochs "$LOCAL_EPOCHS" \
    --fraction-fit "$FRACTION_FIT" \
    --train-split "$TRAIN_SPLIT" \
    --prox-mu "$PROX_MU" \
    --dataset-config-path "$ELSA_DATASET_CONFIG_PATH" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --run-tag "$RUN_TAG" \
    --server-device "cuda:0" \
    --client-device "cuda:0" \
    --num-clients 400 \
    --client-num-cpus "$CLIENT_NUM_CPUS" \
    --client-num-gpus "$CLIENT_NUM_GPUS" \
    --ray-num-cpus "$RAY_NUM_CPUS" \
    --ray-num-gpus 1 \
    --ray-temp-dir "${RAY_TMP_ROOT}/g${GPU_INDEX}" \
    --summary-path "$RESULTS_ROOT/${TASK_NAME}_${RUN_TAG}.json" \
    2>&1 | tee -a "$RUN_LOG"
  RUN_RC=${PIPESTATUS[0]}
  set -e

  echo "[$(date --iso-8601=seconds)] python runner exit_code=${RUN_RC}" | tee -a "$RUN_LOG"
  if [[ "$RUN_RC" -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] task ${TASK_NAME} completed successfully" | tee -a "$RUN_LOG"
    break
  fi

  echo "[$(date --iso-8601=seconds)] task ${TASK_NAME} failed, relaunching in ${RETRY_DELAY_SEC}s" | tee -a "$RUN_LOG"
  sleep "$RETRY_DELAY_SEC"
done
