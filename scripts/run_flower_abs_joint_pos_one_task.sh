#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/ensure_local_xvfb.sh"

TASK_NAME="${1:?task name required}"
GPU_INDEX="${2:?gpu index required}"
NUM_ROUNDS="${3:-3}"
LOCAL_EPOCHS="${4:-5}"
RUN_TAG="${5:-abs-jpos-flwr-smoke-v1}"
FRACTION_FIT="${6:-0.05}"
TRAIN_SPLIT="${7:-0.9}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-60}"
LOG_ROOT="${REPO_ROOT}/logs/flower_abs_joint_pos"

mkdir -p "$LOG_ROOT"

RUN_LOG="${LOG_ROOT}/${TASK_NAME}_${RUN_TAG}.log"
GPU_DEVICE="cuda:${GPU_INDEX}"

cd "$REPO_ROOT"

wait_for_run_completion() {
  local run_id="$1"
  local state_db="/home/cvlab-dgx/.flwr/local-superlink/state.db"
  local signed_run_id="$run_id"
  signed_run_id="$(python - <<PY
run_id = int("${run_id}")
if run_id >= 2**63:
    run_id -= 2**64
print(run_id)
PY
)"
  while true; do
    if [[ ! -f "$state_db" ]]; then
      echo "[$(date --iso-8601=seconds)] waiting for state db ${state_db}" | tee -a "$RUN_LOG"
      sleep "$CHECK_INTERVAL_SEC"
      continue
    fi

    local row
    row="$(sqlite3 "$state_db" "select coalesce(finished_at,''), coalesce(sub_status,''), coalesce(details,'') from run where run_id=${signed_run_id};")"
    if [[ -z "$row" ]]; then
      echo "[$(date --iso-8601=seconds)] waiting for run ${run_id} row (sqlite id ${signed_run_id})" | tee -a "$RUN_LOG"
      sleep "$CHECK_INTERVAL_SEC"
      continue
    fi

    local finished_at sub_status details
    finished_at="${row%%|*}"
    row="${row#*|}"
    sub_status="${row%%|*}"
    details="${row#*|}"

    if [[ -n "$finished_at" ]]; then
      echo "[$(date --iso-8601=seconds)] run ${run_id} finished sub_status='${sub_status}' details='${details}'" | tee -a "$RUN_LOG"
      if [[ "$sub_status" == "failed" ]] || [[ "$details" == *"No heartbeat"* ]]; then
        return 1
      fi
      return 0
    fi

    echo "[$(date --iso-8601=seconds)] run ${run_id} still running" | tee -a "$RUN_LOG"
    sleep "$CHECK_INTERVAL_SEC"
  done
}

while true; do
  echo "[$(date --iso-8601=seconds)] launching task=${TASK_NAME} rounds=${NUM_ROUNDS} local_epochs=${LOCAL_EPOCHS} gpu=${GPU_DEVICE}" | tee -a "$RUN_LOG"

  set +e
  RUN_OUTPUT="$(
    conda run -n "$ELSA_ENV_NAME" flwr run . \
      --run-config "dataset-task='${TASK_NAME}' num-server-rounds=${NUM_ROUNDS} local-epochs=${LOCAL_EPOCHS} fraction-fit=${FRACTION_FIT} fraction-eval=0.0 train-split=${TRAIN_SPLIT} use-wandb=false server-device='${GPU_DEVICE}' client-device='${GPU_DEVICE}' run-tag='${RUN_TAG}' checkpoint-root='model_checkpoints' dataset-config-path='dataset_config.yaml'" \
      2>&1
  )"
  RUN_RC=$?
  set -e

  echo "$RUN_OUTPUT" | tee -a "$RUN_LOG"
  echo "[$(date --iso-8601=seconds)] flwr run exit_code=${RUN_RC}" | tee -a "$RUN_LOG"

  RUN_ID="$(echo "$RUN_OUTPUT" | awk '/Successfully started run/{print $NF}' | tail -n 1)"
  if [[ -z "$RUN_ID" ]]; then
    echo "[$(date --iso-8601=seconds)] failed to parse run id, retrying in ${RETRY_DELAY_SEC}s" | tee -a "$RUN_LOG"
    sleep "$RETRY_DELAY_SEC"
    continue
  fi

  if wait_for_run_completion "$RUN_ID"; then
    echo "[$(date --iso-8601=seconds)] task ${TASK_NAME} completed successfully" | tee -a "$RUN_LOG"
    break
  fi

  echo "[$(date --iso-8601=seconds)] task ${TASK_NAME} failed, relaunching in ${RETRY_DELAY_SEC}s" | tee -a "$RUN_LOG"
  sleep "$RETRY_DELAY_SEC"
done
