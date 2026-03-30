#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <task> <round> [local_epochs]"
  exit 1
fi

TASK="$1"
ROUND="$2"
LOCAL_EPOCHS="${3:-25}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common_env.sh"

MODEL_PATH="$ELSA_ROOT/model_checkpoints/$TASK/fedavg_FKCameraObjectPolicy_l-ep_${LOCAL_EPOCHS}_ts_0.9_fclients_0.05_round_${ROUND}.pth"
OUT_PATH="$ELSA_ROOT/results/live_eval/${TASK}_round_${ROUND}.json"

"$(cd "$(dirname "$0")" && pwd)/run_eval_checkpoint_online.sh" \
  "$MODEL_PATH" \
  "$TASK" \
  "$OUT_PATH" \
  "eval"
