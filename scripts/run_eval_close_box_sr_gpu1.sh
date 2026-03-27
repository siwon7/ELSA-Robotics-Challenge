#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate elsa-robotics-challenge

export CUDA_VISIBLE_DEVICES=1
export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1

cd /home/cv25/siwon/ELSA-Robotics-Challenge

xvfb-run -a python -m federated_elsa_robotics.eval_model \
  --task close_box \
  --local_epochs 50 \
  --fraction_fit 0.05 \
  --train_test_split 0.9 \
  --round 28 \
  --split eval \
  --device cuda:0 \
  --num_workers 4 \
  --simulator \
  --plotting
