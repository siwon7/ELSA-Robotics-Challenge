"""Shared online evaluation utilities for live RLBench rollouts.

This module intentionally keeps a small API surface. Older Ray-based batch
evaluation and plotting code was removed because it duplicated newer scripts
and had diverged from the current action-pipeline semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from elsa_learning_agent.utils import (
    denormalize_action,
    execute_action_with_adapter,
    expand_action_bounds,
    load_environment,
    process_obs,
    select_receding_horizon_actions,
)


def _resolve_collection_config_path(base_cfg) -> Path:
    task_name = str(getattr(base_cfg.dataset, "task", "") or getattr(base_cfg.env, "task_name", ""))
    if not task_name:
        raise ValueError("Could not resolve task name for live evaluation.")
    root_dir = Path(str(base_cfg.dataset.root_dir))
    return root_dir / task_name / f"{task_name}_fed.json"


def online_evaluation(
    agent,
    device,
    transform,
    base_cfg,
    idx_environment: int,
    num_episodes: int = 5,
    max_steps: int = 300,
) -> list[float]:
    """Run live RLBench evaluation and return per-episode rewards."""
    agent.eval()

    collection_cfg_path = _resolve_collection_config_path(base_cfg)
    with collection_cfg_path.open("r", encoding="utf-8") as fh:
        collection_cfg = json.load(fh)

    task_env, rlbench_env = load_environment(
        base_cfg,
        collection_cfg,
        idx_environment,
        headless=True,
    )

    rewards: list[float] = []
    try:
        for _ in range(num_episodes):
            _descriptions, obs = task_env.reset()
            reward = 0.0
            terminated = False
            steps = 0

            while not terminated and steps < max_steps:
                front_rgb, low_dim_state = process_obs(obs, transform)
                front_rgb = front_rgb.unsqueeze(0).to(device)
                low_dim_state = low_dim_state.unsqueeze(0).to(device)

                with torch.no_grad():
                    action = agent.get_action(front_rgb, low_dim_state)

                action_min, action_max = expand_action_bounds(
                    base_cfg.transform.action_min,
                    base_cfg.transform.action_max,
                    int(action.shape[-1]),
                )
                denormalized_action = denormalize_action(
                    action.detach().cpu(),
                    action_min,
                    action_max,
                )
                env_actions = select_receding_horizon_actions(denormalized_action, base_cfg)

                for env_action in env_actions:
                    obs, reward, terminated, executed_steps, _frames = execute_action_with_adapter(
                        task_env,
                        obs,
                        env_action.numpy()[0],
                        base_cfg,
                    )
                    steps += int(executed_steps)
                    if terminated or steps >= max_steps:
                        break

            rewards.append(float(reward))
    finally:
        rlbench_env.shutdown()

    return rewards
