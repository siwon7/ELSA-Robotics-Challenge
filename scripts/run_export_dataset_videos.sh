#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <root_dir> <task> <env_id> <output_dir> [max_episodes] [fps] [episode_idx ...]"
  exit 1
fi

ROOT_DIR="$1"
TASK="$2"
ENV_ID="$3"
OUTPUT_DIR="$4"
MAX_EPISODES="${5:-3}"
FPS="${6:-20}"

cd "$(cd "$(dirname "$0")/.." && pwd)"

EPISODE_ARGS=()
if [ "$#" -gt 6 ]; then
  EPISODE_ARGS=(--episode-indices "${@:7}")
fi

if [ -x ".venv/bin/python" ]; then
  .venv/bin/python scripts/export_dataset_videos.py \
    --root-dir "$ROOT_DIR" \
    --task "$TASK" \
    --env-id "$ENV_ID" \
    --output-dir "$OUTPUT_DIR" \
    --max-episodes "$MAX_EPISODES" \
    --fps "$FPS" \
    "${EPISODE_ARGS[@]}"
else
  python scripts/export_dataset_videos.py \
    --root-dir "$ROOT_DIR" \
    --task "$TASK" \
    --env-id "$ENV_ID" \
    --output-dir "$OUTPUT_DIR" \
    --max-episodes "$MAX_EPISODES" \
    --fps "$FPS" \
    "${EPISODE_ARGS[@]}"
fi
