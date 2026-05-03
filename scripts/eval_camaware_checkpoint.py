"""Eval-only rerun for camera-aware checkpoints.

Loads the checkpoint saved at the end of training (after the train script's
own eval crashed on dim mismatch) and re-runs online_evaluation with the
fixed inference path (ELSA_INCLUDE_CAMERA_IN_STATE=1 must be exported).

Mirrors the second half of scripts/train_same_env_bcpolicy_probe.py.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.utils import (
    get_action_chunk_len,
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
    get_receding_horizon_execute_steps,
)
from federated_elsa_robotics.eval_model import online_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-config-path", required=True)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--train-env-ids", nargs="*", type=int, default=None)
    parser.add_argument("--eval-env-ids", nargs="*", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--output-root",
        default="/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/results/same_env_bcpolicy_probes",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/model_checkpoints/same_env_bcpolicy_probes",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_env_ids = (
        [int(env_id) for env_id in args.train_env_ids]
        if args.train_env_ids
        else [int(args.env_id)]
    )
    eval_env_ids = (
        [int(env_id) for env_id in args.eval_env_ids]
        if args.eval_env_ids
        else list(train_env_ids)
    )

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    cfg = OmegaConf.load(args.dataset_config_path)
    cfg.dataset.task = args.task
    cfg.dataset.env_id = int(train_env_ids[0])
    cfg.dataset.env_ids = list(train_env_ids)

    # Avoid double-concatenation of (K, T) into low_dim_state. With the
    # ELSA_INCLUDE_CAMERA_IN_STATE=1 env var, process_obs already appends
    # (K, T); the dataset_loader's include_camera_in_state flag would then
    # append them a second time. We turn the loader-side flag off here so
    # the resulting low_dim_state matches the 33-dim that the train-time
    # checkpoint expects.
    cfg.dataset.include_camera_in_state = False
    cfg.dataset.train_split = 0.05
    cfg.dataset.test_split = 0.05
    cfg.dataset.batch_size = 1
    cfg.dataset.num_workers = 0
    probe_dataset = ImitationDataset(cfg, train=True, normalize=True)
    probe_loader = DataLoader(probe_dataset, batch_size=1, shuffle=False)
    sample = next(iter(probe_loader))
    print(f"sample low_dim_state shape: {tuple(sample['low_dim_state'].shape)}", flush=True)

    agent = Agent(
        image_channels=sample["image"].shape[1],
        low_dim_state_dim=sample["low_dim_state"].shape[1],
        action_dim=sample["action"].shape[1],
        image_size=(sample["image"].shape[2], sample["image"].shape[3]),
        **get_agent_model_kwargs(cfg),
    )

    env_group_label = (
        f"env_{train_env_ids[0]:03d}"
        if len(train_env_ids) == 1
        else "train_envs_" + "_".join(f"{env_id:03d}" for env_id in train_env_ids)
    )
    ckpt_path = Path(args.checkpoint_root) / args.task / args.run_name / f"{env_group_label}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    print(f"loading checkpoint: {ckpt_path}", flush=True)
    agent.load_state_dict(str(ckpt_path))
    agent.policy.to(device)
    agent.policy.eval()

    base_cfg = OmegaConf.load(
        os.path.join(cfg.dataset.root_dir, cfg.dataset.task, f"{cfg.dataset.task}_fed.yaml")
    )
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform

    per_env_rewards = {}
    per_env_sr = {}
    flat_rewards = []
    eval_start = time.perf_counter()
    for eval_env_id in eval_env_ids:
        rewards = online_evaluation(
            agent,
            device,
            get_image_transform(cfg),
            base_cfg,
            eval_env_id,
            num_episodes=args.eval_episodes,
        )
        rewards = [float(x) for x in rewards]
        per_env_rewards[str(eval_env_id)] = rewards
        per_env_sr[str(eval_env_id)] = float(np.mean(rewards)) if rewards else None
        flat_rewards.extend(rewards)
    sr = float(np.mean(flat_rewards)) if flat_rewards else None
    std_sr = float(np.std(flat_rewards)) if flat_rewards else None
    per_env_sr_values = [value for value in per_env_sr.values() if value is not None]
    eval_elapsed = time.perf_counter() - eval_start

    result_dir = Path(args.output_root) / args.task / args.run_name / env_group_label
    result_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "task": args.task,
        "train_env_ids": train_env_ids,
        "eval_env_ids": eval_env_ids,
        "run_name": args.run_name,
        "policy_name": "BCPolicy",
        "checkpoint_path": str(ckpt_path),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "action_chunk_len": int(get_action_chunk_len(cfg)),
        "receding_horizon_execute_steps": int(get_receding_horizon_execute_steps(cfg)),
        "eval": {
            "rewards_per_env": per_env_rewards,
            "per_env_sr": per_env_sr,
            "mean_per_env_sr": float(np.mean(per_env_sr_values)) if per_env_sr_values else None,
            "std_per_env_sr": float(np.std(per_env_sr_values)) if per_env_sr_values else None,
            "sr": sr,
            "std_sr": std_sr,
            "wall_time_sec": float(eval_elapsed),
        },
        "sr": sr,
        "std_sr": std_sr,
        "per_env_sr": per_env_sr,
        "mean_per_env_sr": float(np.mean(per_env_sr_values)) if per_env_sr_values else None,
        "std_per_env_sr": float(np.std(per_env_sr_values)) if per_env_sr_values else None,
        "include_camera_in_state": bool(
            getattr(cfg.dataset, "include_camera_in_state", False)
        ),
        "env_var_camera": os.getenv("ELSA_INCLUDE_CAMERA_IN_STATE", "0"),
    }
    out_path = result_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.task}][{args.run_name}] sr={sr:.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
