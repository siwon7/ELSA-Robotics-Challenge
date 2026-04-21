import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import (
    BASE_DATASET_CONFIG_PATH,
    get_agent_model_kwargs,
    infer_checkpoint_config_path,
    load_runtime_config,
)
from elsa_learning_agent.live_rollout import (
    load_task_environment,
    rollout_episode,
    save_gif,
)
from elsa_learning_agent.utils import (
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
)
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-dir", required=True)
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

    base_cfg = OmegaConf.load(f"{root_dir}/{args.task}/{args.task}_fed.yaml")
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform

    agent = Agent(
        image_channels=3,
        low_dim_state_dim=8,
        action_dim=int(infer_action_dim(cfg)),
        image_size=(128, 128),
        **get_agent_model_kwargs(cfg),
    )
    state_dict = torch.load(args.model_path, map_location=torch.device(args.device))
    agent.policy.load_state_dict(state_dict)
    agent.policy.to(args.device)
    agent.eval()

    transform = get_image_transform(cfg)
    action_min = torch.tensor(base_cfg.transform.action_min)
    action_max = torch.tensor(base_cfg.transform.action_max)
    video_root = Path(args.video_dir)
    video_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    success_flags = []
    results = []
    for env_id in env_ids:
        task_env, rlbench_env = load_task_environment(base_cfg, env_id, headless=True)
        try:
            env_video_dir = video_root / f"env_{env_id:03d}"
            env_video_dir.mkdir(parents=True, exist_ok=True)
            env_runs = []
            for episode_idx in range(args.episodes):
                episode = rollout_episode(
                    agent,
                    task_env,
                    transform,
                    args.device,
                    action_min,
                    action_max,
                    args.max_steps,
                    base_cfg,
                    capture_frames=True,
                )
                video_path = env_video_dir / f"episode_{episode_idx:03d}.gif"
                save_gif(video_path, episode["frames"], fps=args.fps)
                item = {
                    "env_id": env_id,
                    "episode_idx": episode_idx,
                    "reward": episode["reward"],
                    "terminated": episode["terminated"],
                    "success": episode["success"],
                    "steps": episode["steps"],
                    "video_path": str(video_path),
                }
                env_runs.append(item)
                results.append(item)
                success_flags.append(1.0 if episode["success"] else 0.0)
        finally:
            rlbench_env.shutdown()

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
        "sr": float(np.mean(success_flags)) if success_flags else 0.0,
        "std_sr": float(np.std(success_flags)) if success_flags else 0.0,
        "num_rollouts": len(success_flags),
        "elapsed_sec": float(time.perf_counter() - start),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
