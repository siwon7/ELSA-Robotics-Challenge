#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import (
    BASE_DATASET_CONFIG_PATH,
    get_agent_model_kwargs,
    infer_checkpoint_config_path,
    load_runtime_config,
)
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.live_rollout import load_task_environment, rollout_episode, save_gif
from elsa_learning_agent.utils import (
    denormalize_action,
    expand_action_bounds,
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
    get_receding_horizon_execute_steps,
    move_nested_to_device,
    process_obs,
    process_obs_with_context,
    requires_observation_context,
    select_receding_horizon_actions,
)
from federated_elsa_robotics.task import infer_action_dim, validate_one_epoch

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


def tensor_stats(x: torch.Tensor) -> dict:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def build_agent(cfg, sample, device: torch.device) -> Agent:
    agent = Agent(
        image_channels=3,
        low_dim_state_dim=int(sample["low_dim_state"].shape[1]),
        action_dim=int(sample["action"].shape[1]),
        image_size=(int(sample["image"].shape[2]), int(sample["image"].shape[3])),
        **get_agent_model_kwargs(cfg),
    )
    agent.policy.to(device)
    return agent


def load_agent_from_checkpoint(cfg, model_path: str, sample, device: torch.device) -> Agent:
    agent = build_agent(cfg, sample, device)
    state_dict = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(state_dict)
    agent.eval()
    return agent


def compute_image_usage(agent, loader, device, num_batches: int) -> dict:
    zero_image_deltas = []
    shuffle_image_deltas = []
    zero_state_deltas = []
    shuffle_state_deltas = []
    base_pred_std = []
    saturation_fractions = []
    image_embed_norms = []
    state_embed_norms = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break
            image = batch["image"].to(device)
            low_dim = batch["low_dim_state"].to(device)
            obs_context = move_nested_to_device(batch.get("obs_context"), device)

            pred = agent.policy(image, low_dim, obs_context=obs_context)
            pred_zero_image = agent.policy(torch.zeros_like(image), low_dim, obs_context=obs_context)
            pred_zero_state = agent.policy(image, torch.zeros_like(low_dim), obs_context=obs_context)

            if image.shape[0] > 1:
                image_perm = torch.randperm(image.shape[0], device=device)
                low_dim_perm = torch.randperm(low_dim.shape[0], device=device)
                shuffled_context = None
                if isinstance(obs_context, dict):
                    shuffled_context = {
                        key: value[image_perm] if torch.is_tensor(value) and value.shape[0] == image.shape[0] else value
                        for key, value in obs_context.items()
                    }
                pred_shuffle_image = agent.policy(image[image_perm], low_dim, obs_context=shuffled_context or obs_context)
                pred_shuffle_state = agent.policy(image, low_dim[low_dim_perm], obs_context=obs_context)
            else:
                pred_shuffle_image = pred.clone()
                pred_shuffle_state = pred.clone()

            zero_image_deltas.append(torch.norm(pred - pred_zero_image, dim=-1).mean().item())
            shuffle_image_deltas.append(torch.norm(pred - pred_shuffle_image, dim=-1).mean().item())
            zero_state_deltas.append(torch.norm(pred - pred_zero_state, dim=-1).mean().item())
            shuffle_state_deltas.append(torch.norm(pred - pred_shuffle_state, dim=-1).mean().item())
            base_pred_std.append(pred.std(dim=0, unbiased=False).mean().item())
            saturation_fractions.append((pred.abs() > 0.99).float().mean().item())

            image_embed = agent.policy.cnn_encoder(image, obs_context) if requires_observation_context(loader.dataset.config) else agent.policy.cnn_encoder(image)
            if isinstance(image_embed, dict):
                image_embed = image_embed["global_embedding"]
            state_embed = agent.policy.mlp_encoder(low_dim)
            image_embed_norms.append(image_embed.norm(dim=-1).mean().item())
            state_embed_norms.append(state_embed.norm(dim=-1).mean().item())

    return {
        "base_prediction": tensor_stats(torch.tensor(base_pred_std)),
        "zero_image_l2_delta": tensor_stats(torch.tensor(zero_image_deltas)),
        "shuffle_image_l2_delta": tensor_stats(torch.tensor(shuffle_image_deltas)),
        "zero_state_l2_delta": tensor_stats(torch.tensor(zero_state_deltas)),
        "shuffle_state_l2_delta": tensor_stats(torch.tensor(shuffle_state_deltas)),
        "saturation_fraction": tensor_stats(torch.tensor(saturation_fractions)),
        "image_embedding_norm": tensor_stats(torch.tensor(image_embed_norms)),
        "state_embedding_norm": tensor_stats(torch.tensor(state_embed_norms)),
    }


def collect_initial_actions(
    agent,
    task_env,
    transform,
    device,
    action_min,
    action_max,
    cfg,
    episodes: int,
) -> dict:
    actions = []
    with torch.no_grad():
        for episode_idx in range(episodes):
            _descriptions, obs = task_env.reset()
            if requires_observation_context(cfg):
                front_rgb, low_dim_state, obs_context = process_obs_with_context(obs, transform)
                obs_context = {
                    key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim >= 1 else value
                    for key, value in obs_context.items()
                }
                obs_context = move_nested_to_device(obs_context, device)
            else:
                front_rgb, low_dim_state = process_obs(obs, transform)
                obs_context = None
            front_rgb = front_rgb.unsqueeze(0).to(device)
            low_dim_state = low_dim_state.unsqueeze(0).to(device)

            pred = agent.get_action(front_rgb, low_dim_state, obs_context=obs_context)
            expanded_action_min, expanded_action_max = expand_action_bounds(
                action_min,
                action_max,
                int(pred.shape[-1]),
            )
            denorm = denormalize_action(pred.detach().cpu(), expanded_action_min, expanded_action_max)
            env_action = select_receding_horizon_actions(denorm, cfg)[0].numpy()[0]
            actions.append(
                {
                    "episode_idx": episode_idx,
                    "env_action": env_action.tolist(),
                    "raw_pred": pred.detach().cpu().numpy()[0].tolist(),
                }
            )

    action_array = np.asarray([item["env_action"] for item in actions], dtype=np.float32)
    return {
        "actions": actions,
        "mean": action_array.mean(axis=0).tolist(),
        "std": action_array.std(axis=0).tolist(),
        "mean_l2_std": float(np.linalg.norm(action_array.std(axis=0))),
        "max_std": float(action_array.std(axis=0).max()),
    }


def evaluate_with_execute_steps(
    agent,
    base_cfg,
    env_id: int,
    transform,
    device,
    action_min,
    action_max,
    execute_steps: int,
    episodes: int,
    max_steps: int,
) -> dict:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    cfg.dataset.receding_horizon_execute_steps = int(execute_steps)

    task_env, rlbench_env = load_task_environment(cfg, env_id, headless=True)
    try:
        rewards = []
        for _ in range(episodes):
            episode = rollout_episode(
                agent,
                task_env,
                transform,
                device,
                action_min,
                action_max,
                max_steps,
                cfg,
            )
            rewards.append(float(episode["reward"]))
    finally:
        rlbench_env.shutdown()

    return {
        "rewards": rewards,
        "sr": float(np.mean(rewards)) if rewards else 0.0,
    }


def build_interpretation(summary: dict) -> dict:
    image_metrics = summary.get("image_usage", {})
    initial_metrics = summary.get("initial_action", {})
    execute_metrics = summary.get("execute_steps_sweep", {})
    online_metrics = summary.get("online_seen_env", {})
    offline_metrics = summary.get("offline_seen_env", {})

    zero_image = float(image_metrics.get("zero_image_l2_delta", {}).get("mean", 0.0))
    zero_state = float(image_metrics.get("zero_state_l2_delta", {}).get("mean", 0.0))
    saturation = float(image_metrics.get("saturation_fraction", {}).get("mean", 0.0))
    max_std = float(initial_metrics.get("max_std", 0.0))
    execute_one = float(execute_metrics.get("execute_1", {}).get("sr", 0.0))
    best_execute = max(
        (float(v.get("sr", 0.0)) for v in execute_metrics.values()),
        default=0.0,
    )
    online_sr = float(online_metrics.get("mean_reward", 0.0) or 0.0)
    offline_rmse = float(offline_metrics.get("rmse", 0.0) or 0.0)

    return {
        "vision_path_alive": zero_image > 1.0e-3,
        "state_dominant": zero_state > max(zero_image * 2.0, 1.0e-3),
        "prediction_saturated": saturation > 0.25,
        "initial_action_collapsed": max_std < 0.02,
        "execute_steps_sensitive": (best_execute - execute_one) > 0.1,
        "closed_loop_gap": offline_rmse < 0.1 and online_sr <= 0.0,
    }


def parse_execute_steps(raw: str, chunk_len: int) -> list[int]:
    if not raw:
        return [1]
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        return [1]
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return [max(1, min(chunk_len, value)) for value in deduped]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--dataset-config-path", default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-usage-batches", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--initial-action-episodes", type=int, default=5)
    parser.add_argument("--execute-steps", default="")
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    resolved_config_path = (
        Path(args.dataset_config_path)
        if args.dataset_config_path is not None
        else infer_checkpoint_config_path(args.model_path)
    )
    cfg = load_runtime_config(
        resolved_config_path if resolved_config_path is not None else BASE_DATASET_CONFIG_PATH,
        task=args.task,
        env_id=args.env_id,
    )
    if args.train_split is not None:
        cfg.dataset.train_split = float(args.train_split)
    if "train_split" not in cfg.dataset:
        cfg.dataset.train_split = 0.9
    cfg.dataset.test_split = float(cfg.dataset.train_split)
    cfg.dataset.batch_size = int(args.batch_size)
    cfg.dataset.num_workers = int(args.num_workers)
    cfg.dataset.action_dim = infer_action_dim(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = ImitationDataset(cfg, train=True, normalize=True)
    val_dataset = ImitationDataset(cfg, test=True, normalize=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    sample = next(iter(val_loader))
    agent = load_agent_from_checkpoint(cfg, args.model_path, sample, device)

    offline_loss = validate_one_epoch(agent, val_loader, device)
    transform = get_image_transform(cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(
        os.path.join(cfg.dataset.root_dir, cfg.dataset.task, f"{cfg.dataset.task}_fed.yaml")
    )
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform

    task_env, rlbench_env = load_task_environment(base_cfg, args.env_id, headless=True)
    try:
        action_min = torch.tensor(base_cfg.transform.action_min)
        action_max = torch.tensor(base_cfg.transform.action_max)

        start = time.perf_counter()
        rewards = []
        for _ in range(args.eval_episodes):
            episode = rollout_episode(
                agent,
                task_env,
                transform,
                device,
                action_min,
                action_max,
                args.max_steps,
                base_cfg,
            )
            rewards.append(float(episode["reward"]))
        online_payload = {
            "rewards": rewards,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "elapsed_sec": float(time.perf_counter() - start),
        }

        initial_action_payload = collect_initial_actions(
            agent,
            task_env,
            transform,
            device,
            action_min,
            action_max,
            base_cfg,
            episodes=args.initial_action_episodes,
        )
    finally:
        rlbench_env.shutdown()

    image_usage_payload = {
        "model_path": args.model_path,
        "dataset_config_path": str(
            resolved_config_path if resolved_config_path is not None else BASE_DATASET_CONFIG_PATH
        ),
        "task": args.task,
        "env_id": int(args.env_id),
        "split": "test",
        "num_samples": int(min(len(val_dataset), args.batch_size * args.image_usage_batches)),
        "vision_backbone": str(getattr(cfg.model, "vision_backbone", "cnn")),
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "action_output_activation": get_agent_model_kwargs(cfg)["action_output_activation"],
        "metrics": compute_image_usage(agent, val_loader, device, args.image_usage_batches),
    }

    execute_steps_candidates = parse_execute_steps(
        args.execute_steps,
        int(getattr(cfg.dataset, "action_chunk_len", 1) or 1),
    )
    execute_steps_payload = {}
    for execute_steps in execute_steps_candidates:
        execute_steps_payload[f"execute_{execute_steps}"] = evaluate_with_execute_steps(
            agent,
            base_cfg,
            env_id=args.env_id,
            transform=transform,
            device=device,
            action_min=action_min,
            action_max=action_max,
            execute_steps=execute_steps,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
        )

    video_payload = None
    if args.video_episodes > 0:
        task_env, rlbench_env = load_task_environment(base_cfg, args.env_id, headless=True)
        try:
            video_root = output_dir / f"videos_env{args.env_id}_{args.video_episodes}ep"
            env_video_dir = video_root / f"env_{args.env_id:03d}"
            env_video_dir.mkdir(parents=True, exist_ok=True)
            video_results = []
            for episode_idx in range(args.video_episodes):
                episode = rollout_episode(
                    agent,
                    task_env,
                    transform,
                    device,
                    action_min,
                    action_max,
                    args.max_steps,
                    base_cfg,
                    capture_frames=True,
                )
                video_path = env_video_dir / f"episode_{episode_idx:03d}.gif"
                save_gif(video_path, episode["frames"], fps=args.fps)
                video_results.append(
                    {
                        "episode_idx": episode_idx,
                        "reward": float(episode["reward"]),
                        "success": bool(episode["success"]),
                        "steps": int(episode["steps"]),
                        "video_path": str(video_path),
                    }
                )
        finally:
            rlbench_env.shutdown()
        video_payload = {"results": video_results}

    summary = {
        "task": args.task,
        "env_id": int(args.env_id),
        "model_path": args.model_path,
        "resolved_config_path": str(
            resolved_config_path if resolved_config_path is not None else BASE_DATASET_CONFIG_PATH
        ),
        "vision_backbone": str(getattr(cfg.model, "vision_backbone", "cnn")),
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "default_execute_steps": int(get_receding_horizon_execute_steps(cfg)),
        "offline_seen_env": {
            "mean_loss": float(offline_loss),
            "rmse": float(math.sqrt(offline_loss)),
        },
        "online_seen_env": online_payload,
        "image_usage": image_usage_payload["metrics"],
        "initial_action": initial_action_payload,
        "execute_steps_sweep": execute_steps_payload,
    }
    summary["diagnostic_flags"] = build_interpretation(summary)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "image_usage.json").write_text(
        json.dumps(image_usage_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "initial_actions.json").write_text(
        json.dumps(initial_action_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "execute_steps_eval.json").write_text(
        json.dumps(execute_steps_payload, indent=2), encoding="utf-8"
    )
    if video_payload is not None:
        (output_dir / "video_eval.json").write_text(
            json.dumps(video_payload, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
