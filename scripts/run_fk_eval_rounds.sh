#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <task> <gpu> <rounds_csv>"
  exit 1
fi

TASK="$1"
GPU="$2"
ROUNDS_CSV="$3"

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

mkdir -p "$ROOT/logs/fk_eval" "$ROOT/results/fk_eval"
cd "$ROOT"

IFS=',' read -r -a ROUNDS <<< "$ROUNDS_CSV"

for r in "${ROUNDS[@]}"; do
  echo "=== ${TASK} round ${r} ===" | tee -a "logs/fk_eval/${TASK}.log"
  xvfb-run -a python scripts/eval_checkpoint.py \
    --model-path "model_checkpoints/${TASK}/fedavg_FKBCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_${r}.pth" \
    --task "$TASK" \
    --device cuda:0 \
    --simulator \
    --output "results/fk_eval/${TASK}_round_${r}.json" \
    2>&1 | tee -a "logs/fk_eval/${TASK}.log"
done
