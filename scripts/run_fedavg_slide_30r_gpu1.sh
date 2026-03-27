#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate elsa-robotics-challenge

export CUDA_VISIBLE_DEVICES=1

cd /home/cv25/siwon/ELSA-Robotics-Challenge

flwr run . --stream \
  -c 'dataset-task="slide_block_to_target" use-wandb=false server-device="cuda:0" num-server-rounds=30 local-epochs=50 fraction-fit=0.05 train-split=0.9'
