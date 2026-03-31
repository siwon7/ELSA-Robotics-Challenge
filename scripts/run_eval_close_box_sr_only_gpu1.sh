#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${ELSA_TASK:-close_box}"
ROUND="${ELSA_ROUND:-28}"
LOCAL_EPOCHS="${ELSA_LOCAL_EPOCHS:-50}"
TRAIN_TEST_SPLIT="${ELSA_TRAIN_TEST_SPLIT:-0.9}"
FRACTION_FIT="${ELSA_FRACTION_FIT:-0.05}"
SPLIT="${ELSA_SPLIT:-eval}"
OUTPUT_JSON="${ELSA_OUTPUT_JSON:-$ROOT_DIR/results/live_eval/${TASK}_round_${ROUND}.online.${SPLIT}.json}"

export CUDA_VISIBLE_DEVICES="${ELSA_CUDA_VISIBLE_DEVICES:-1}"

"$SCRIPT_DIR/run_eval_checkpoint_online.sh" \
  "model_checkpoints/${TASK}/BCPolicy_l-ep_${LOCAL_EPOCHS}_ts_${TRAIN_TEST_SPLIT}_fclients_${FRACTION_FIT}_round_${ROUND}.pth" \
  "$TASK" \
  "$OUTPUT_JSON" \
  "$SPLIT"
