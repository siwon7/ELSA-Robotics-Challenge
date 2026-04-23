#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conservative defaults to reduce host pressure.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
export TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-1}"
export EVAL_EPISODES="${EVAL_EPISODES:-3}"

exec "$SCRIPT_DIR/run_phase1_central400_unseen.sh" "$@"
