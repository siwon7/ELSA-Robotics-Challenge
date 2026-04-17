#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <model_path> <task> <output_json> [split]"
  exit 1
fi

MODEL_PATH="$1"
TASK="$2"
OUTPUT_JSON="$3"
SPLIT="${4:-eval}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/prepare_live_eval_env.sh"

mkdir -p "$(dirname "$OUTPUT_JSON")"
cd "$ELSA_ROOT"

EXTRA_ARGS=()
if [ -n "${ELSA_POLICY_NAME:-}" ]; then
  EXTRA_ARGS+=(--policy-name "$ELSA_POLICY_NAME")
fi
if [ -n "${ELSA_DATASET_CONFIG_PATH:-}" ]; then
  EXTRA_ARGS+=(--dataset-config-path "$ELSA_DATASET_CONFIG_PATH")
fi
if [ -n "${ELSA_OFFLINE_ENV_START:-}" ] && [ -n "${ELSA_OFFLINE_ENV_END:-}" ]; then
  EXTRA_ARGS+=(--offline-env-start "$ELSA_OFFLINE_ENV_START" --offline-env-end "$ELSA_OFFLINE_ENV_END")
fi
if [ -n "${ELSA_LIVE_ENV_IDS:-}" ]; then
  EXTRA_ARGS+=(--live-env-ids "$ELSA_LIVE_ENV_IDS")
fi
if [ -n "${ELSA_NUM_EPISODES_LIVE:-}" ]; then
  EXTRA_ARGS+=(--num-episodes-live "$ELSA_NUM_EPISODES_LIVE")
fi

if [ -x "$ELSA_VENV_PATH/bin/python" ]; then
  "$ELSA_VENV_PATH/bin/python" scripts/eval_checkpoint.py \
    --model-path "$MODEL_PATH" \
    --task "$TASK" \
    --device "${ELSA_SIM_DEVICE:-cpu}" \
    --split "$SPLIT" \
    "${EXTRA_ARGS[@]}" \
    --simulator \
    --output "$OUTPUT_JSON"
else
  conda run -n "$ELSA_ENV_NAME" \
    python scripts/eval_checkpoint.py \
    --model-path "$MODEL_PATH" \
    --task "$TASK" \
    --device "${ELSA_SIM_DEVICE:-cpu}" \
    --split "$SPLIT" \
    "${EXTRA_ARGS[@]}" \
    --simulator \
    --output "$OUTPUT_JSON"
fi
