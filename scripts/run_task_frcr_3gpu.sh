#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ELSA_POLICY_NAME="${ELSA_POLICY_NAME:-frcr}"
export ELSA_DETERMINISTIC_TRAINING="${ELSA_DETERMINISTIC_TRAINING:-false}"

exec "$SCRIPT_DIR/run_task_direct_3gpu.sh"
