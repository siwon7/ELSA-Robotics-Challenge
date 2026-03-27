#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <task> <round> <gpu>"
  exit 1
fi

TASK="$1"
ROUND="$2"
GPU="$3"

ROOT="/home/cv25/siwon/ELSA-Robotics-Challenge"
COPPELIASIM_ROOT="/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate elsa-robotics-challenge

export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR=/tmp/matplotlib
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1

cd "$ROOT"
mkdir -p logs/fk_eval results/fk_eval

LOG_PATH="logs/fk_eval/${TASK}_round_${ROUND}.log"
OUT_PATH="results/fk_eval/${TASK}_round_${ROUND}.json"

echo "=== ${TASK} round ${ROUND} ===" | tee -a "$LOG_PATH"
xvfb-run -a python scripts/eval_checkpoint.py \
  --model-path "model_checkpoints/${TASK}/fedavg_FKBCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_${ROUND}.pth" \
  --task "$TASK" \
  --device cuda:0 \
  --simulator \
  --output "$OUT_PATH" \
  2>&1 | tee -a "$LOG_PATH"
