#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ELSA_ROUNDS="${ELSA_ROUNDS:-20,21,22,23,24,25,26,27}" \
  "$SCRIPT_DIR/run_eval_close_box_sr_sweep_gpu1.sh"
