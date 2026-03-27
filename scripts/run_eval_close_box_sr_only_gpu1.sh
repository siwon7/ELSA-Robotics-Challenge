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

xvfb-run -a python - <<'PY'
import json
from omegaconf import OmegaConf
from federated_elsa_robotics.eval_model import load_agent, evaluate_online

cfg = OmegaConf.load("dataset_config.yaml")
agent = load_agent(
    "model_checkpoints/close_box/BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_28.pth",
    "cuda:0",
)
online = evaluate_online(
    agent=agent,
    base_config=cfg,
    task="close_box",
    split="eval",
    device="cuda:0",
)
print(json.dumps(online, indent=2))
PY
