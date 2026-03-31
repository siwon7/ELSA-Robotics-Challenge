#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/live_eval"

export ELSA_SIM_HEADLESS="${ELSA_SIM_HEADLESS:-1}"
export ELSA_SIM_DEVICE="${ELSA_SIM_DEVICE:-cpu}"

MODELS=(
  "slide_block_to_target|model_checkpoints/slide_block_to_target/BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth|$RESULT_DIR/slide_block_to_target_round_30.online.test.compat_v3.json"
  "close_box|model_checkpoints/close_box/BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_17.pth|$RESULT_DIR/close_box_round_17.online.test.compat_v3.json"
)

mkdir -p "$RESULT_DIR"
cd "$ROOT_DIR"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r task model_path output_json <<<"$entry"
  echo "=== START $task ==="
  "$SCRIPT_DIR/run_eval_checkpoint_online.sh" "$model_path" "$task" "$output_json" test
  echo "=== END $task ==="
done
