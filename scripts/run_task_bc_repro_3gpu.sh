#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ELSA_POLICY_NAME="${ELSA_POLICY_NAME:-legacy_bc}"
export ELSA_SEED="${ELSA_SEED:-0}"
export ELSA_DETERMINISTIC_TRAINING="${ELSA_DETERMINISTIC_TRAINING:-true}"
export ELSA_USE_WANDB="${ELSA_USE_WANDB:-false}"

exec "$SCRIPT_DIR/run_task_3gpu.sh"
