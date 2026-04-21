import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from colosseum import TASKS_PY_FOLDER, TASKS_TTM_FOLDER
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import JointPosition, JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.utils import (
    denormalize_action,
    get_image_transform,
    load_environment,
    process_obs,
)
from scripts.live_eval_common import load_main_split_cfg


class MoveArmThenGripperJointPosition(ActionMode):
    def action(self, scene, action: np.ndarray):
        arm_act_size = int(np.prod(self.arm_action_mode.action_shape(scene)))
        arm_action = np.array(action[:arm_act_size], dtype=np.float32)
        gripper_action = np.array(
            action[arm_act_size : arm_act_size + 1], dtype=np.float32
        )
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, gripper_action)

    def action_shape(self, scene):
        return int(np.prod(self.arm_action_mode.action_shape(scene))) + int(
            np.prod(self.gripper_action_mode.action_shape(scene))
        )


def parse_env_ids(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("env ids are empty")
    return values


def load_joint_position_environment(base_cfg, collection_cfg, idx_environment: int, headless: bool = True):
    task = name_to_class(base_cfg.env.task_name, TASKS_PY_FOLDER)
    config = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    env_entry = next(
        entry for entry in collection_cfg["env_config"] if entry["env_idx"] == idx_environment
    )
    for variation_cfg in env_entry.get("variations_parameters", []):
        var_type = variation_cfg["type"]
        var_name = variation_cfg.get("name")
        for factor_cfg in config.env.scene.factors:
            if factor_cfg.variation != var_type:
                continue
            if var_name is not None and "name" in factor_cfg and factor_cfg.name != var_name:
                continue
            for key, value in variation_cfg.items():
                if key == "type":
                    continue
                factor_cfg[key] = value
            break

    obs_config = ObservationConfigExt(config.data)
    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripperJointPosition(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=headless,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=config.env,
    )
    rlbench_env.launch()
    task_env = rlbench_env.get_task(task)
    return task_env, rlbench_env


def load_agent(model_path: str, device: str, cfg):
    agent = Agent(
        image_channels=3,
        low_dim_state_dim=8,
        action_dim=8,
        image_size=(128, 128),
        **get_agent_model_kwargs(cfg),
    )
    state_dict = torch.load(model_path, map_location=torch.device(device))
    agent.policy.load_state_dict(state_dict)
    agent.policy.to(device)
    agent.eval()
    return agent


def run_eval_episode(agent, task_env, transform, device: str, mode: str, dt: float, max_steps: int):
    _descriptions, obs = task_env.reset()
    reward = 0.0
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        front_rgb, low_dim_state = process_obs(obs, transform)
        front_rgb = front_rgb.unsqueeze(0).to(device)
        low_dim_state = low_dim_state.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = agent.get_action(front_rgb, low_dim_state)
        action = denormalize_action(
            pred.detach().cpu(),
            torch.tensor([-1.0] * 7 + [0.0]),
            torch.tensor([1.0] * 8),
        ).numpy()[0]

        if mode == "joint_velocity":
            env_action = action
        elif mode == "joint_position_integrated":
            q = np.asarray(obs.joint_positions, dtype=np.float32)
            next_q = q + np.asarray(action[:7], dtype=np.float32) * float(dt)
            env_action = np.concatenate((next_q, np.asarray([action[7]], dtype=np.float32)), axis=0)
        else:
            raise ValueError(f"unsupported mode: {mode}")

        obs, step_reward, terminate = task_env.step(env_action)
        reward = float(step_reward)
        terminated = bool(terminate)
        steps += 1
    return {
        "reward": reward,
        "terminated": terminated,
        "success": bool(terminated or reward > 0.0),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--env-ids", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--mode", required=True, choices=["joint_velocity", "joint_position_integrated"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env_ids = parse_env_ids(args.env_ids)
    cfg, collection_cfg = load_main_split_cfg(args.task, args.split)
    transform = get_image_transform(cfg)
    agent = load_agent(args.model_path, args.device, cfg)

    start = time.perf_counter()
    results = []
    success_flags = []
    for env_id in env_ids:
        if args.mode == "joint_velocity":
            task_env, rlbench_env = load_environment(cfg, collection_cfg, env_id, headless=True)
        else:
            task_env, rlbench_env = load_joint_position_environment(cfg, collection_cfg, env_id, headless=True)
        try:
            env_runs = []
            for episode_idx in range(args.episodes):
                item = run_eval_episode(
                    agent, task_env, transform, args.device, args.mode, args.dt, args.max_steps
                )
                item["episode_idx"] = episode_idx
                env_runs.append(item)
                success_flags.append(1.0 if item["success"] else 0.0)
            results.append({"env_id": env_id, "episodes": env_runs})
        finally:
            rlbench_env.shutdown()

    payload = {
        "model_path": args.model_path,
        "task": args.task,
        "split": args.split,
        "env_ids": env_ids,
        "episodes": args.episodes,
        "mode": args.mode,
        "dt": args.dt,
        "sr": float(np.mean(success_flags)) if success_flags else 0.0,
        "elapsed_sec": float(time.perf_counter() - start),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
