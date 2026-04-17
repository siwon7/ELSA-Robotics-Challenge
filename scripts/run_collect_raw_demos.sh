#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 7 ]; then
  echo "usage: $0 <task> <env_id> <output_dir> [split] [num_episodes] [base_seed] [headless_0_or_1]"
  exit 1
fi

TASK="$1"
ENV_ID="$2"
OUTPUT_DIR="$3"
SPLIT="${4:-eval}"
NUM_EPISODES="${5:-5}"
BASE_SEED="${6:-12345}"
HEADLESS_FLAG="${7:-0}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/prepare_live_eval_env.sh"

cd "$ELSA_ROOT"

HEADLESS_ARGS=()
if [ "$HEADLESS_FLAG" = "1" ]; then
  HEADLESS_ARGS+=(--headless)
fi

if [ -x "$ELSA_VENV_PATH/bin/python" ]; then
  "$ELSA_VENV_PATH/bin/python" scripts/collect_raw_demos.py \
    --task "$TASK" \
    --env-id "$ENV_ID" \
    --split "$SPLIT" \
    --num-episodes "$NUM_EPISODES" \
    --base-seed "$BASE_SEED" \
    --output-dir "$OUTPUT_DIR" \
    "${HEADLESS_ARGS[@]}"
else
  conda run -n "$ELSA_ENV_NAME" \
    python scripts/collect_raw_demos.py \
    --task "$TASK" \
    --env-id "$ENV_ID" \
    --split "$SPLIT" \
    --num-episodes "$NUM_EPISODES" \
    --base-seed "$BASE_SEED" \
    --output-dir "$OUTPUT_DIR" \
    "${HEADLESS_ARGS[@]}"
fi
