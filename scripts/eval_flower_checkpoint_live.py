import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import (
    BASE_DATASET_CONFIG_PATH,
    infer_checkpoint_config_path,
    load_runtime_config,
)
from elsa_learning_agent.utils import (
    get_action_output_activation,
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
)
from federated_elsa_robotics.eval_model import online_evaluation
from federated_elsa_robotics.task import infer_action_dim

try:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


def resolve_split(cfg, split: str) -> tuple[str, list[int]]:
    if split == "eval":
        return str(cfg.dataset.root_eval_dir), list(cfg.dataset.final_eval_live_idxs)
    if split == "test":
        return str(cfg.dataset.root_test_dir), list(cfg.dataset.final_test_live_idxs)
    if split == "training":
        return str(cfg.dataset.root_dir), [int(cfg.dataset.env_id)]
    raise ValueError(f"Unsupported split: {split}")


def parse_env_ids(raw: str | None, default_env_ids: list[int]) -> list[int]:
    if raw is None:
        return default_env_ids
    env_ids = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            env_ids.append(int(token))
    if not env_ids:
        raise ValueError("env ids are empty")
    return env_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-config-path", default=None)
    parser.add_argument("--split", default="eval", choices=["training", "eval", "test"])
    parser.add_argument("--env-ids", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    resolved_config_path = (
        Path(args.dataset_config_path)
        if args.dataset_config_path is not None
        else infer_checkpoint_config_path(args.model_path)
    )
    cfg = load_runtime_config(
        resolved_config_path if resolved_config_path is not None else BASE_DATASET_CONFIG_PATH,
        task=args.task,
    )
    root_dir, default_env_ids = resolve_split(cfg, args.split)
    env_ids = parse_env_ids(args.env_ids, default_env_ids)
    cfg.dataset.root_dir = root_dir
    cfg.dataset.enable_live_eval = True
    cfg.dataset.num_episodes_live = int(args.episodes)

    base_cfg = OmegaConf.load(f"{root_dir}/{args.task}/{args.task}_fed.yaml")
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform

    agent = Agent(
        image_channels=3,
        low_dim_state_dim=8,
        action_dim=int(infer_action_dim(cfg)),
        image_size=(128, 128),
        vision_backbone=str(getattr(cfg.model, "vision_backbone", "cnn")),
        projector_dim=int(getattr(cfg.model, "projector_dim", 256)),
        action_output_activation=get_action_output_activation(cfg),
        normalize_branch_embeddings=bool(
            getattr(cfg.model, "normalize_branch_embeddings", False)
        ),
        low_dim_dropout_prob=float(
            getattr(cfg.model, "low_dim_dropout_prob", 0.0) or 0.0
        ),
    )
    state_dict = torch.load(args.model_path, map_location=torch.device(args.device))
    agent.policy.load_state_dict(state_dict)
    agent.policy.to(args.device)
    agent.eval()

    start = time.perf_counter()
    per_env = {}
    all_rewards = []
    transform = get_image_transform(cfg)
    for env_id in env_ids:
        rewards = online_evaluation(
            agent,
            args.device,
            transform,
            base_cfg,
            env_id,
            num_episodes=args.episodes,
        )
        per_env[str(env_id)] = [float(x) for x in rewards]
        all_rewards.extend(float(x) for x in rewards)

    payload = {
        "model_path": args.model_path,
        "task": args.task,
        "resolved_config_path": str(
            resolved_config_path if resolved_config_path is not None else BASE_DATASET_CONFIG_PATH
        ),
        "split": args.split,
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "episodes_per_env": int(args.episodes),
        "env_ids": env_ids,
        "sr": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_sr": float(np.std(all_rewards)) if all_rewards else 0.0,
        "num_rollouts": len(all_rewards),
        "per_env_rewards": per_env,
        "elapsed_sec": float(time.perf_counter() - start),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
