"""Shared online evaluation utilities for live RLBench rollouts.

This module intentionally keeps a small API surface. Older Ray-based batch
evaluation and plotting code was removed because it duplicated newer scripts
and had diverged from the current action-pipeline semantics.
"""

from __future__ import annotations

from elsa_learning_agent.live_rollout import load_task_environment, rollout_episode


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
    task_env, rlbench_env = load_task_environment(base_cfg, idx_environment, headless=True)

    rewards: list[float] = []
    try:
        for _ in range(num_episodes):
            episode = rollout_episode(
                agent,
                task_env,
                transform,
                device,
                base_cfg.transform.action_min,
                base_cfg.transform.action_max,
                max_steps,
                base_cfg,
                capture_frames=False,
            )
            rewards.append(float(episode["reward"]))
    finally:
        rlbench_env.shutdown()

    return rewards
