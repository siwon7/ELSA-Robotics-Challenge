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
from pathlib import Path

from omegaconf import OmegaConf

from federated_elsa_robotics.eval_model import evaluate_online, load_agent

rounds = [28, 30, 40, 50, 58]
cfg = OmegaConf.load("dataset_config.yaml")
results = []

for round_num in rounds:
    ckpt = Path(
        f"model_checkpoints/close_box/BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_{round_num}.pth"
    )
    print(f"=== round {round_num} ===", flush=True)
    agent = load_agent(str(ckpt), "cuda:0")
    online = evaluate_online(
        agent=agent,
        base_config=cfg,
        task="close_box",
        split="eval",
        device="cuda:0",
    )
    result = {
        "round": round_num,
        "mean_reward": online["mean_reward"],
        "std_reward": online["std_reward"],
        "rewards_per_env": online["rewards_per_env"],
    }
    results.append(result)
    print(json.dumps(result, indent=2), flush=True)

out_dir = Path("results/close_box_sr_sweep")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rounds_28_30_40_50_58.json"
out_path.write_text(json.dumps(results, indent=2))
best = max(results, key=lambda item: item["mean_reward"])
print("=== best ===", flush=True)
print(json.dumps(best, indent=2), flush=True)
print(f"saved_to={out_path}", flush=True)
PY
