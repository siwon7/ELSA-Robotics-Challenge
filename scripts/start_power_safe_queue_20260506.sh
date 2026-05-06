#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"

# This mode is for the current unstable DGX state: run one training worker at a
# time and keep CPU-side pressure low.  It does not hard-cap GPU watts unless an
# administrator has already applied nvidia-smi power limits.
export ELSA_FILL3_LOG_ROOT="${ELSA_FILL3_LOG_ROOT:-$ARTIFACT_ROOT/logs/power_safe_20260506}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-2}"
export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
export ELSA_DATALOADER_WORKERS="${ELSA_DATALOADER_WORKERS:-0}"
export NUM_WORKERS="${NUM_WORKERS:-$ELSA_DATALOADER_WORKERS}"
export POLL_SEC="${POLL_SEC:-60}"

exec bash "$SCRIPT_DIR/run_cpu_limited_fill3_queues_20260506.sh"
