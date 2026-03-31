#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 <model_path> <task> <gpu> <output_json> [split]"
  exit 1
fi

MODEL_PATH="$1"
TASK="$2"
GPU="$3"
OUTPUT_JSON="$4"
SPLIT="${5:-eval}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common_env.sh"

mkdir -p "$(dirname "$OUTPUT_JSON")"
cd "$ELSA_ROOT"
EVAL_BATCH_SIZE="${ELSA_EVAL_BATCH_SIZE:-32}"
EVAL_NUM_WORKERS="${ELSA_EVAL_NUM_WORKERS:-0}"
EVAL_DEVICE="${ELSA_EVAL_DEVICE:-cuda:0}"

if [ "$EVAL_DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

EXTRA_ARGS=()
if [ -n "${ELSA_POLICY_NAME:-}" ]; then
  EXTRA_ARGS+=(--policy-name "$ELSA_POLICY_NAME")
fi

if [ -x "$ELSA_VENV_PATH/bin/python" ]; then
  "$ELSA_VENV_PATH/bin/python" scripts/eval_checkpoint.py \
    --model-path "$MODEL_PATH" \
    --task "$TASK" \
    --device "$EVAL_DEVICE" \
    --split "$SPLIT" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$EVAL_NUM_WORKERS" \
    "${EXTRA_ARGS[@]}" \
    --output "$OUTPUT_JSON"
else
  conda run -n "$ELSA_ENV_NAME" \
    python scripts/eval_checkpoint.py \
    --model-path "$MODEL_PATH" \
    --task "$TASK" \
    --device "$EVAL_DEVICE" \
    --split "$SPLIT" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$EVAL_NUM_WORKERS" \
    "${EXTRA_ARGS[@]}" \
    --output "$OUTPUT_JSON"
fi
