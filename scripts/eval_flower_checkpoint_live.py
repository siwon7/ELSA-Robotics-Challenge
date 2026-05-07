import argparse
import json
import os
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
from elsa_learning_agent.utils import (
    get_expected_image_channels,
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
)
from federated_elsa_robotics.eval_model import online_evaluation
from federated_elsa_robotics.task import infer_action_dim, infer_low_dim_state_dim

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
    parser.add_argument("--action-pipeline-preset", default=None)
    parser.add_argument("--execution-action-interface", default=None)
    parser.add_argument("--execution-action-adapter", default=None)
    parser.add_argument("--receding-horizon-execute-steps", type=int, default=None)
    parser.add_argument("--joint-velocity-servo-gain", type=float, default=None)
    parser.add_argument("--joint-velocity-servo-clip", type=float, default=None)
    parser.add_argument("--joint-velocity-servo-steps", type=int, default=None)
    parser.add_argument("--joint-velocity-servo-tolerance", type=float, default=None)
    parser.add_argument(
        "--gripper-eval-mode",
        choices=["threshold", "hysteresis"],
        default=None,
    )
    parser.add_argument("--gripper-open-threshold", type=float, default=None)
    parser.add_argument("--gripper-close-threshold", type=float, default=None)
    parser.add_argument("--gripper-min-hold-steps", type=int, default=None)
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
    if bool(getattr(cfg.dataset, "include_camera_in_state", False)):
        os.environ["ELSA_INCLUDE_CAMERA_IN_STATE"] = "1"
    if args.action_pipeline_preset is not None:
        cfg.dataset.action_pipeline_preset = args.action_pipeline_preset
    if args.execution_action_interface is not None:
        cfg.dataset.execution_action_interface = args.execution_action_interface
    if args.execution_action_adapter is not None:
        cfg.dataset.execution_action_adapter = args.execution_action_adapter
    if args.receding_horizon_execute_steps is not None:
        cfg.dataset.receding_horizon_execute_steps = int(
            args.receding_horizon_execute_steps
        )
    if args.joint_velocity_servo_gain is not None:
        cfg.dataset.joint_velocity_servo_gain = float(args.joint_velocity_servo_gain)
    if args.joint_velocity_servo_clip is not None:
        cfg.dataset.joint_velocity_servo_clip = float(args.joint_velocity_servo_clip)
    if args.joint_velocity_servo_steps is not None:
        cfg.dataset.joint_velocity_servo_steps = int(args.joint_velocity_servo_steps)
    if args.joint_velocity_servo_tolerance is not None:
        cfg.dataset.joint_velocity_servo_tolerance = float(
            args.joint_velocity_servo_tolerance
        )
    if args.gripper_eval_mode is not None:
        cfg.dataset.gripper_eval_mode = args.gripper_eval_mode
    if args.gripper_open_threshold is not None:
        cfg.dataset.gripper_open_threshold = float(args.gripper_open_threshold)
    if args.gripper_close_threshold is not None:
        cfg.dataset.gripper_close_threshold = float(args.gripper_close_threshold)
    if args.gripper_min_hold_steps is not None:
        cfg.dataset.gripper_min_hold_steps = int(args.gripper_min_hold_steps)

    base_cfg = OmegaConf.load(f"{root_dir}/{args.task}/{args.task}_fed.yaml")
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform

    agent = Agent(
        image_channels=get_expected_image_channels(cfg),
        low_dim_state_dim=infer_low_dim_state_dim(cfg),
        action_dim=int(infer_action_dim(cfg)),
        image_size=(128, 128),
        **get_agent_model_kwargs(cfg),
    )
    state_dict = torch.load(args.model_path, map_location=torch.device(args.device))
    agent.policy.load_state_dict(state_dict)
    agent.policy.to(args.device)
    agent.eval()

    start = time.perf_counter()
    per_env = {}
    per_env_episode_metrics = {}
    all_rewards = []
    all_predicted_flips = []
    all_executed_flips = []
    transform = get_image_transform(cfg)
    for env_id in env_ids:
        episodes = online_evaluation(
            agent,
            args.device,
            transform,
            base_cfg,
            env_id,
            num_episodes=args.episodes,
            return_episodes=True,
        )
        rewards = [float(episode["reward"]) for episode in episodes]
        per_env[str(env_id)] = [float(x) for x in rewards]
        all_rewards.extend(float(x) for x in rewards)
        per_env_episode_metrics[str(env_id)] = [
            {
                "reward": float(episode["reward"]),
                "success": bool(episode["success"]),
                "steps": int(episode["steps"]),
                "gripper_eval_mode": str(episode.get("gripper_eval_mode", "threshold")),
                "gripper_hysteresis_applied": bool(
                    episode.get("gripper_hysteresis_applied", False)
                ),
                "predicted_gripper_flips": int(
                    episode.get("predicted_gripper_flips", 0)
                ),
                "executed_gripper_flips": int(
                    episode.get("executed_gripper_flips", 0)
                ),
            }
            for episode in episodes
        ]
        all_predicted_flips.extend(
            int(episode.get("predicted_gripper_flips", 0)) for episode in episodes
        )
        all_executed_flips.extend(
            int(episode.get("executed_gripper_flips", 0)) for episode in episodes
        )

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
        "gripper_eval_mode": str(
            getattr(cfg.dataset, "gripper_eval_mode", "threshold") or "threshold"
        ),
        "gripper_open_threshold": float(
            getattr(cfg.dataset, "gripper_open_threshold", 0.65) or 0.65
        ),
        "gripper_close_threshold": float(
            getattr(cfg.dataset, "gripper_close_threshold", 0.35) or 0.35
        ),
        "gripper_min_hold_steps": int(
            getattr(cfg.dataset, "gripper_min_hold_steps", 0) or 0
        ),
        "episodes_per_env": int(args.episodes),
        "env_ids": env_ids,
        "sr": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_sr": float(np.std(all_rewards)) if all_rewards else 0.0,
        "num_rollouts": len(all_rewards),
        "per_env_rewards": per_env,
        "per_env_episode_metrics": per_env_episode_metrics,
        "mean_predicted_gripper_flips": (
            float(np.mean(all_predicted_flips)) if all_predicted_flips else 0.0
        ),
        "mean_executed_gripper_flips": (
            float(np.mean(all_executed_flips)) if all_executed_flips else 0.0
        ),
        "elapsed_sec": float(time.perf_counter() - start),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
