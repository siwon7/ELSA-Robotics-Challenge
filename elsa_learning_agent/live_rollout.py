from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency fallback
    imageio = None

from elsa_learning_agent.utils import (
    denormalize_action,
    execute_action_with_adapter,
    expand_action_bounds,
    load_environment,
    move_nested_to_device,
    process_obs,
    process_obs_with_context,
    requires_observation_context,
    select_receding_horizon_actions,
)


def resolve_collection_config_path(base_cfg) -> Path:
    task_name = str(getattr(base_cfg.dataset, "task", "") or getattr(base_cfg.env, "task_name", ""))
    if not task_name:
        raise ValueError("Could not resolve task name for live evaluation.")
    root_dir = Path(str(base_cfg.dataset.root_dir))
    return root_dir / task_name / f"{task_name}_fed.json"


def load_task_environment(base_cfg, env_id: int, *, headless: bool = True):
    collection_cfg_path = resolve_collection_config_path(base_cfg)
    with collection_cfg_path.open("r", encoding="utf-8") as fh:
        collection_cfg = json.load(fh)
    return load_environment(base_cfg, collection_cfg, env_id, headless=headless)


def rollout_episode(
    agent,
    task_env,
    transform,
    device,
    action_min,
    action_max,
    max_steps: int,
    cfg,
    *,
    capture_frames: bool = False,
):
    _descriptions, obs = task_env.reset()
    frames = [np.asarray(obs.front_rgb, dtype=np.uint8)] if capture_frames else []
    reward = 0.0
    terminated = False
    steps = 0
    needs_obs_context = requires_observation_context(cfg) or str(
        getattr(getattr(agent, "policy", None), "vision_backbone", "") or ""
    ) == "volumedp_lite_dinov3_vits16"

    while not terminated and steps < max_steps:
        if needs_obs_context:
            front_rgb, low_dim_state, obs_context = process_obs_with_context(obs, transform)
            obs_context = {
                key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim >= 1 else value
                for key, value in obs_context.items()
            }
        else:
            front_rgb, low_dim_state = process_obs(obs, transform)
            obs_context = None
        front_rgb = front_rgb.unsqueeze(0).to(device)
        low_dim_state = low_dim_state.unsqueeze(0).to(device)
        obs_context = move_nested_to_device(obs_context, device)

        with torch.no_grad():
            action = agent.get_action(front_rgb, low_dim_state, obs_context=obs_context)
        expanded_action_min, expanded_action_max = expand_action_bounds(
            action_min,
            action_max,
            int(action.shape[-1]),
        )
        denormalized_action = denormalize_action(
            action.detach().cpu(), expanded_action_min, expanded_action_max
        )
        env_actions = select_receding_horizon_actions(denormalized_action, cfg)
        for env_action in env_actions:
            obs, step_reward, terminate, executed_steps, step_frames = execute_action_with_adapter(
                task_env,
                obs,
                env_action.numpy()[0],
                cfg,
            )
            if capture_frames:
                frames.extend(step_frames)
            reward = float(step_reward)
            terminated = bool(terminate)
            steps += int(executed_steps)
            if terminated or steps >= max_steps:
                break

    success = bool(terminated or reward > 0.0)
    return {
        "reward": reward,
        "terminated": terminated,
        "success": success,
        "steps": steps,
        "frames": frames,
    }


def save_gif(path: Path, frames, fps: int):
    if imageio is not None:
        imageio.mimsave(path, frames, fps=fps)
        return

    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
