#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${ELSA_TASK:-close_box}"
ROUNDS_CSV="${ELSA_ROUNDS:-28,30,40,50,58}"
LOCAL_EPOCHS="${ELSA_LOCAL_EPOCHS:-50}"
TRAIN_TEST_SPLIT="${ELSA_TRAIN_TEST_SPLIT:-0.9}"
FRACTION_FIT="${ELSA_FRACTION_FIT:-0.05}"
SPLIT="${ELSA_SPLIT:-eval}"
OUT_DIR="${ELSA_OUTPUT_DIR:-$ROOT_DIR/results/${TASK}_sr_sweep}"

export CUDA_VISIBLE_DEVICES="${ELSA_CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$OUT_DIR"

IFS=',' read -r -a ROUNDS <<<"$ROUNDS_CSV"
RESULT_FILES=()

for round_num in "${ROUNDS[@]}"; do
  output_json="$OUT_DIR/${TASK}_round_${round_num}.online.${SPLIT}.json"
  "$SCRIPT_DIR/run_eval_checkpoint_online.sh" \
    "model_checkpoints/${TASK}/BCPolicy_l-ep_${LOCAL_EPOCHS}_ts_${TRAIN_TEST_SPLIT}_fclients_${FRACTION_FIT}_round_${round_num}.pth" \
    "$TASK" \
    "$output_json" \
    "$SPLIT"
  RESULT_FILES+=("$output_json")
done

summary_json="$OUT_DIR/rounds_${ROUNDS_CSV//,/_}.json"

"$ROOT_DIR/.venv/bin/python" - "${RESULT_FILES[@]}" "$summary_json" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:-1]]
out_path = Path(sys.argv[-1])
results = [json.loads(path.read_text()) for path in paths]
best = max(results, key=lambda item: item.get("sr", 0.0))
out_path.write_text(json.dumps({"results": results, "best": best}, indent=2))
print(json.dumps(best, indent=2))
print(f"saved_to={out_path}")
PY
