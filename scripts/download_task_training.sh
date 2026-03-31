#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ELSA_TASK="${ELSA_TASK:-slide_block_to_target}"
export ELSA_STORAGE_ROOT="${ELSA_STORAGE_ROOT:-/mnt/ext_sdb1/elsa_robotics_challenge}"
export ELSA_DATA_ROOT="${ELSA_DATA_ROOT:-$ELSA_STORAGE_ROOT/datasets}"
export ELSA_TRAIN_ENVS="${ELSA_TRAIN_ENVS:-400}"

mkdir -p "$ELSA_DATA_ROOT" "$ELSA_STORAGE_ROOT/logs"

cd "$ROOT_DIR"
source "$ROOT_DIR/.venv/bin/activate"
python -u data_downloader.py --data_type training --task "$ELSA_TASK" --num_envs "$ELSA_TRAIN_ENVS"
