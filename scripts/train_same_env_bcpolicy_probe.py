import argparse
import json
import math
import os
import random
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
from federated_elsa_robotics.task import train, validate_one_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--dataset-config-path",
        default="/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/dataset_config.yaml",
    )
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default="bcpolicy_same_env_probe")
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

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    cfg = OmegaConf.load(args.dataset_config_path)
    cfg.dataset.task = args.task
    cfg.dataset.env_id = int(args.env_id)
    cfg.dataset.train_split = float(args.train_split)
    cfg.dataset.test_split = float(args.train_split)
    cfg.dataset.batch_size = int(args.batch_size)
    cfg.dataset.num_workers = int(args.num_workers)

    train_dataset = ImitationDataset(cfg, train=True, normalize=True)
    val_dataset = ImitationDataset(cfg, test=True, normalize=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample = next(iter(train_loader))
    agent = Agent(
        image_channels=3,
        low_dim_state_dim=sample["low_dim_state"].shape[1],
        action_dim=sample["action"].shape[1],
        image_size=(sample["image"].shape[2], sample["image"].shape[3]),
        **get_agent_model_kwargs(cfg),
    )

    history = []
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        loss = train(agent, train_loader, 1, device, cfg)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(loss),
                "elapsed_sec": float(time.perf_counter() - train_start),
            }
        )
        print(
            f"[train] task={args.task} epoch={epoch + 1}/{args.epochs} "
            f"mean_train_loss={loss:.6f}",
            flush=True,
        )

    result_dir = Path(args.output_root) / args.task / args.run_name / f"env_{args.env_id:03d}"
    ckpt_dir = Path(args.checkpoint_root) / args.task / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"env_{args.env_id:03d}.pth"
    resolved_config_path = result_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_config_path)

    offline_loss = validate_one_epoch(agent, val_loader, device)
    agent.save(str(ckpt_path))

    base_cfg = OmegaConf.load(
        os.path.join(cfg.dataset.root_dir, cfg.dataset.task, f"{cfg.dataset.task}_fed.yaml")
    )
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform

    rewards = online_evaluation(
        agent,
        device,
        get_image_transform(cfg),
        base_cfg,
        args.env_id,
        num_episodes=args.eval_episodes,
    )
    sr = float(np.mean(rewards)) if rewards else None
    std_sr = float(np.std(rewards)) if rewards else None

    result = {
        "task": args.task,
        "train_env_ids": [args.env_id],
        "eval_env_ids": [args.env_id],
        "epochs": args.epochs,
        "train_split": args.train_split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "policy_name": "BCPolicy",
        "resolved_config_path": str(resolved_config_path),
        "vision_backbone": str(getattr(cfg.model, "vision_backbone", "cnn")),
        "projector_dim": int(getattr(cfg.model, "projector_dim", 256)),
        "policy_head_type": str(getattr(cfg.model, "policy_head_type", "mlp")),
        "diffusion_num_steps": int(getattr(cfg.model, "diffusion_num_steps", 20) or 20),
        "diffusion_hidden_dim": int(
            getattr(cfg.model, "diffusion_hidden_dim", 512) or 512
        ),
        "diffusion_timestep_dim": int(
            getattr(cfg.model, "diffusion_timestep_dim", 128) or 128
        ),
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "action_chunk_len": int(get_action_chunk_len(cfg)),
        "action_keyframe_horizon": int(
            getattr(cfg.dataset, "action_keyframe_horizon", 1) or 1
        ),
        "receding_horizon_execute_steps": int(get_receding_horizon_execute_steps(cfg)),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "checkpoint_path": str(ckpt_path),
        "train": {
            "history": history,
            "timing": {"wall_time_sec": float(time.perf_counter() - train_start)},
        },
        "offline_seen_env": {
            "mean_loss": float(offline_loss),
            "rmse": float(math.sqrt(offline_loss)),
        },
        "online_seen_env": {
            "rewards": [float(x) for x in rewards],
            "mean_reward": sr,
            "std_reward": std_sr,
        },
        "per_env_sr": {str(args.env_id): sr},
        "sr": sr,
        "std_sr": std_sr,
    }
    result_path = result_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
