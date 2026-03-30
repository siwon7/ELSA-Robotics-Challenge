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

conda run -n "$ELSA_ENV_NAME" \
  python scripts/eval_checkpoint.py \
  --model-path "$MODEL_PATH" \
  --task "$TASK" \
  --device "${ELSA_SIM_DEVICE:-cpu}" \
  --split "$SPLIT" \
  "${EXTRA_ARGS[@]}" \
  --simulator \
  --output "$OUTPUT_JSON"
