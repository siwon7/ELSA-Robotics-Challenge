#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ELSA_POLICY_NAME="${ELSA_POLICY_NAME:-fk_camera_object}"
export ELSA_DETERMINISTIC_TRAINING="${ELSA_DETERMINISTIC_TRAINING:-false}"

exec "$SCRIPT_DIR/run_task_3gpu.sh"
