#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <task> <round> <gpu> [local_epochs]"
  exit 1
fi

TASK="$1"
ROUND="$2"
GPU="$3"
LOCAL_EPOCHS="${4:-50}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate elsa-robotics-challenge

export CUDA_VISIBLE_DEVICES="$GPU"
export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1

cd /home/cv25/siwon/ELSA-Robotics-Challenge

xvfb-run -a python - <<PY
import json
from pathlib import Path

from omegaconf import OmegaConf

from federated_elsa_robotics.eval_model import evaluate_online, load_agent

task = "${TASK}"
round_num = int("${ROUND}")
local_epochs = int("${LOCAL_EPOCHS}")
cfg = OmegaConf.load("dataset_config.yaml")
ckpt = Path(
    f"model_checkpoints/{task}/BCPolicy_l-ep_{local_epochs}_ts_0.9_fclients_0.05_round_{round_num}.pth"
)
agent = load_agent(str(ckpt), "cuda:0")
online = evaluate_online(
    agent=agent,
    base_config=cfg,
    task=task,
    split="eval",
    device="cuda:0",
)
result = {
    "task": task,
    "round": round_num,
    "local_epochs": local_epochs,
    "mean_reward": online["mean_reward"],
    "std_reward": online["std_reward"],
    "rewards_per_env": online["rewards_per_env"],
}
print(json.dumps(result, indent=2), flush=True)

out_dir = Path("results") / task
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"sr_le{local_epochs}_round_{round_num}.json"
out_path.write_text(json.dumps(result, indent=2))
print(f"saved_to={out_path}", flush=True)
PY
