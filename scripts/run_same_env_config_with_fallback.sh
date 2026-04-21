#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task}"
GPU="${2:?gpu}"
CONFIG_PATH="${3:?config_path}"
RUN_PREFIX="${4:?run_prefix}"
ENV_ID="${5:-0}"
EVAL_EPISODES="${6:-20}"
SEED="${7:-0}"
THRESHOLD="${8:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/same_env_suite"

run_once() {
  local epochs="$1"
  local run_name="${RUN_PREFIX}_e${epochs}_s${SEED}"
  bash "$SCRIPT_DIR/run_same_env_config_one_task.sh" \
    "$TASK" \
    "$GPU" \
    "$CONFIG_PATH" \
    "$epochs" \
    "$ENV_ID" \
    "$run_name" \
    "$EVAL_EPISODES" \
    "$SEED"
}

get_sr() {
  local epochs="$1"
  local run_name="${RUN_PREFIX}_e${epochs}_s${SEED}"
  local result_path="$RESULT_ROOT/$TASK/$run_name/env_$(printf "%03d" "$ENV_ID")/result.json"
  python - "$result_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing result file: {path}")
data = json.loads(path.read_text())
sr = data.get("sr")
if sr is None:
    raise SystemExit(f"missing sr in {path}")
print(float(sr))
PY
}

echo "[runner] task=$TASK gpu=$GPU config=$CONFIG_PATH run_prefix=$RUN_PREFIX env_id=$ENV_ID eval_episodes=$EVAL_EPISODES seed=$SEED threshold=$THRESHOLD"

run_once 50
sr_50="$(get_sr 50)"
echo "[runner] completed 50 epochs with sr=$sr_50"

if python - "$sr_50" "$THRESHOLD" <<'PY'
import sys
sr = float(sys.argv[1])
threshold = float(sys.argv[2])
sys.exit(0 if sr < threshold else 1)
PY
then
  echo "[runner] sr=$sr_50 is below threshold=$THRESHOLD, launching 100 epoch fallback"
  run_once 100
  sr_100="$(get_sr 100)"
  echo "[runner] completed 100 epochs with sr=$sr_100"
else
  echo "[runner] sr=$sr_50 meets threshold=$THRESHOLD, skipping 100 epoch fallback"
fi
