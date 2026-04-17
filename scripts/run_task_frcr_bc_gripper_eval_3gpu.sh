#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ELSA_POLICY_NAME="${ELSA_POLICY_NAME:-frcr_bc_gripper}"
export ELSA_DATASET_CONFIG_PATH="${ELSA_DATASET_CONFIG_PATH:-dataset_config_frcr_bc_gripper.yaml}"
export ELSA_DETERMINISTIC_TRAINING="${ELSA_DETERMINISTIC_TRAINING:-false}"
export ELSA_ENABLE_CLIENT_LOCAL_STATE="${ELSA_ENABLE_CLIENT_LOCAL_STATE:-false}"
export ELSA_ENABLE_CENTRALIZED_EVAL="${ELSA_ENABLE_CENTRALIZED_EVAL:-true}"
export ELSA_CENTRALIZED_EVAL_SIMULATOR="${ELSA_CENTRALIZED_EVAL_SIMULATOR:-false}"

exec "$SCRIPT_DIR/run_task_direct_3gpu.sh"
