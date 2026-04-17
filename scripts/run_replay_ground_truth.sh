#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 11 ]; then
  echo "usage: $0 <task> <env_id> <output_dir> [split] [num_episodes] [save_video_count] [headless_0_or_1] [replay_mode] [raw_demo_dir_or_dash] [reset_mode_or_dash] [continue_after_success_0_or_1]"
  exit 1
fi

TASK="$1"
ENV_ID="$2"
OUTPUT_DIR="$3"
SPLIT="${4:-eval}"
NUM_EPISODES="${5:-5}"
SAVE_VIDEO_COUNT="${6:-1}"
HEADLESS_FLAG="${7:-0}"
REPLAY_MODE="${8:-velocity}"
RAW_DEMO_DIR="${9:--}"
RESET_MODE="${10:--}"
CONTINUE_AFTER_SUCCESS="${11:-0}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/prepare_live_eval_env.sh"

cd "$ELSA_ROOT"

HEADLESS_ARGS=()
if [ "$HEADLESS_FLAG" = "1" ]; then
  HEADLESS_ARGS+=(--headless)
fi

OPTIONAL_ARGS=()
if [ "$RAW_DEMO_DIR" != "-" ]; then
  OPTIONAL_ARGS+=(--raw-demo-dir "$RAW_DEMO_DIR")
fi
if [ "$RESET_MODE" != "-" ]; then
  OPTIONAL_ARGS+=(--reset-mode "$RESET_MODE")
fi
if [ "$CONTINUE_AFTER_SUCCESS" = "1" ]; then
  OPTIONAL_ARGS+=(--continue-after-success)
fi

if [ -x "$ELSA_VENV_PATH/bin/python" ]; then
  "$ELSA_VENV_PATH/bin/python" scripts/replay_ground_truth.py \
    --task "$TASK" \
    --env-id "$ENV_ID" \
    --split "$SPLIT" \
    --num-episodes "$NUM_EPISODES" \
    --save-video-count "$SAVE_VIDEO_COUNT" \
    --max-steps "${ELSA_SIM_MAX_STEPS:-300}" \
    --fps "${ELSA_VIDEO_FPS:-20}" \
    --replay-mode "$REPLAY_MODE" \
    --output-dir "$OUTPUT_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    "${HEADLESS_ARGS[@]}"
else
  conda run -n "$ELSA_ENV_NAME" \
    python scripts/replay_ground_truth.py \
    --task "$TASK" \
    --env-id "$ENV_ID" \
    --split "$SPLIT" \
    --num-episodes "$NUM_EPISODES" \
    --save-video-count "$SAVE_VIDEO_COUNT" \
    --max-steps "${ELSA_SIM_MAX_STEPS:-300}" \
    --fps "${ELSA_VIDEO_FPS:-20}" \
    --replay-mode "$REPLAY_MODE" \
    --output-dir "$OUTPUT_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    "${HEADLESS_ARGS[@]}"
fi
