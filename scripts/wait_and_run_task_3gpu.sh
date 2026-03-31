#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ELSA_TASK="${ELSA_TASK:-slide_block_to_target}"
export ELSA_STORAGE_ROOT="${ELSA_STORAGE_ROOT:-/mnt/ext_sdb1/elsa_robotics_challenge}"
export ELSA_DATA_ROOT="${ELSA_DATA_ROOT:-$ELSA_STORAGE_ROOT/datasets}"
export ELSA_TRAIN_ENVS="${ELSA_TRAIN_ENVS:-400}"
DATA_DIR="$ELSA_DATA_ROOT/training/$ELSA_TASK"
LOG_PATH="${ELSA_TRAIN_WAIT_LOG:-$ELSA_STORAGE_ROOT/logs/run_${ELSA_TASK}_3gpu_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_PATH")"

while true; do
  if [ -d "$DATA_DIR" ]; then
    count=$(find "$DATA_DIR" -maxdepth 1 -type d -name 'env_*' | wc -l)
  else
    count=0
  fi
  echo "[$(date '+%F %T')] $ELSA_TASK training envs: $count/$ELSA_TRAIN_ENVS" | tee -a "$LOG_PATH"
  if [ "$count" -ge "$ELSA_TRAIN_ENVS" ]; then
    break
  fi
  sleep 60
done

cd "$ROOT_DIR"
exec ./scripts/run_task_3gpu.sh |& tee -a "$LOG_PATH"
