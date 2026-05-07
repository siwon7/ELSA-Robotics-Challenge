from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from elsa_learning_agent.config_utils import BASE_DATASET_CONFIG_PATH, load_runtime_config
from elsa_learning_agent.dataset.compat import load_pickled_data
from elsa_learning_agent.dataset.keypoint_discovery import (
    discover_heuristic_keypoints,
    find_next_keypoint_index,
)
from elsa_learning_agent.live_rollout import load_task_environment, rollout_episode
from elsa_learning_agent.utils import (
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
    normalize_action,
)

try:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


@dataclass(frozen=True)
class RetrievalIndex:
    states: torch.Tensor
    phases: torch.Tensor
    actions: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    phase_mean: torch.Tensor
    phase_std: torch.Tensor
    median_demo_steps: int
    demo_count: int
    sample_count: int


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


def resolve_split(cfg, split: str) -> tuple[str, list[int]]:
    if split == "eval":
        return str(cfg.dataset.root_eval_dir), list(cfg.dataset.final_eval_live_idxs)
    if split == "test":
        return str(cfg.dataset.root_test_dir), list(cfg.dataset.final_test_live_idxs)
    if split == "training":
        return str(cfg.dataset.root_dir), [int(cfg.dataset.env_id)]
    raise ValueError(f"Unsupported split: {split}")


def _configured_env_ids(cfg) -> list[int]:
    env_ids_cfg = getattr(cfg.dataset, "env_ids", None)
    if env_ids_cfg:
        return [int(env_id) for env_id in env_ids_cfg]
    return [int(cfg.dataset.env_id)]


def _train_split(cfg) -> float:
    return float(
        getattr(
            cfg.dataset,
            "train_split",
            1.0 - float(getattr(cfg.dataset, "test_split", 0.1)),
        )
    )


def _low_dim_from_obs(obs, cfg) -> np.ndarray:
    low_dim = np.concatenate(
        (
            np.asarray(obs.joint_positions, dtype=np.float32),
            np.asarray([obs.gripper_open], dtype=np.float32),
        ),
        axis=0,
    )
    if bool(getattr(cfg.dataset, "include_camera_in_state", False)):
        misc = getattr(obs, "misc", {}) or {}
        intrinsics = misc.get("front_camera_intrinsics")
        extrinsics = misc.get("front_camera_extrinsics")
        if intrinsics is None or extrinsics is None:
            raise ValueError(
                "include_camera_in_state=True but obs.misc lacks front camera calibration"
            )
        low_dim = np.concatenate(
            (
                low_dim,
                np.asarray(intrinsics, dtype=np.float32).reshape(-1),
                np.asarray(extrinsics, dtype=np.float32).reshape(-1),
            ),
            axis=0,
        )
    return low_dim.astype(np.float32, copy=False)


def _get_keyframe_target_index(cfg, trajectory, time_step: int, keypoints) -> int:
    selection = str(
        getattr(cfg.dataset, "action_keyframe_selection", "fixed_horizon")
    )
    if selection == "fixed_horizon":
        horizon = int(getattr(cfg.dataset, "action_keyframe_horizon", 1) or 1)
        return min(time_step + horizon, len(trajectory) - 1)
    if selection == "peract_heuristic":
        return find_next_keypoint_index(
            keypoints=keypoints or [],
            time_step=time_step,
            fallback_last_index=len(trajectory) - 1,
        )
    raise ValueError(f"Unsupported action_keyframe_selection: {selection}")


def _discover_keypoints_if_needed(cfg, trajectory):
    action_representation = get_action_representation(cfg)
    if action_representation not in {
        "joint_position_keyframe",
        "joint_position_keyframe_relative",
    }:
        return None
    selection = str(
        getattr(cfg.dataset, "action_keyframe_selection", "fixed_horizon")
    )
    if selection == "fixed_horizon":
        return None
    if selection == "peract_heuristic":
        return discover_heuristic_keypoints(
            trajectory,
            stopping_delta=float(
                getattr(cfg.dataset, "action_keyframe_stopping_delta", 0.1)
            ),
            stopped_buffer_steps=int(
                getattr(cfg.dataset, "action_keyframe_stopped_buffer_steps", 4) or 4
            ),
        )
    raise ValueError(f"Unsupported action_keyframe_selection: {selection}")


def _build_single_action(
    cfg,
    trajectory,
    time_step: int,
    *,
    keypoints=None,
    reference_time_step: int | None = None,
) -> np.ndarray:
    clamped_step = min(time_step, len(trajectory) - 2)
    reference_step = (
        clamped_step
        if reference_time_step is None
        else min(reference_time_step, len(trajectory) - 2)
    )
    obs = trajectory[clamped_step]
    next_obs = trajectory[clamped_step + 1]
    reference_obs = trajectory[reference_step]
    action_representation = get_action_representation(cfg)

    if action_representation == "joint_position_absolute":
        arm_action = np.asarray(next_obs.joint_positions, dtype=np.float32)
        gripper_open = next_obs.gripper_open
    elif action_representation == "joint_position_relative":
        arm_action = (
            np.asarray(next_obs.joint_positions, dtype=np.float32)
            - np.asarray(reference_obs.joint_positions, dtype=np.float32)
        )
        gripper_open = next_obs.gripper_open
    elif action_representation == "joint_position_keyframe":
        target_idx = _get_keyframe_target_index(
            cfg, trajectory, clamped_step, keypoints
        )
        target_obs = trajectory[target_idx]
        arm_action = np.asarray(target_obs.joint_positions, dtype=np.float32)
        gripper_open = target_obs.gripper_open
    elif action_representation == "joint_position_keyframe_relative":
        target_idx = _get_keyframe_target_index(
            cfg, trajectory, clamped_step, keypoints
        )
        target_obs = trajectory[target_idx]
        arm_action = (
            np.asarray(target_obs.joint_positions, dtype=np.float32)
            - np.asarray(reference_obs.joint_positions, dtype=np.float32)
        )
        gripper_open = target_obs.gripper_open
    elif action_representation == "joint_velocity":
        arm_action = np.asarray(obs.joint_velocities, dtype=np.float32)
        gripper_open = next_obs.gripper_open
    else:
        raise ValueError(f"Unsupported action_representation: {action_representation}")

    return np.concatenate(
        (arm_action, np.asarray([gripper_open], dtype=np.float32)),
        axis=0,
    ).astype(np.float32, copy=False)


def _build_action(cfg, trajectory, time_step: int, keypoints=None) -> torch.Tensor:
    chunk_len = int(getattr(cfg.dataset, "action_chunk_len", 1) or 1)
    action_seq = [
        _build_single_action(
            cfg,
            trajectory,
            time_step + offset,
            keypoints=keypoints,
            reference_time_step=time_step,
        )
        for offset in range(chunk_len)
    ]
    action = torch.tensor(np.concatenate(action_seq, axis=0), dtype=torch.float32)
    action_min = torch.tensor(cfg.transform.action_min, dtype=torch.float32)
    action_max = torch.tensor(cfg.transform.action_max, dtype=torch.float32)
    if action.numel() % action_min.numel() != 0:
        raise ValueError(
            f"Cannot normalize action dim {action.numel()} with bounds dim {action_min.numel()}"
        )
    repeat_factor = action.numel() // action_min.numel()
    return normalize_action(
        action,
        action_min.repeat(repeat_factor),
        action_max.repeat(repeat_factor),
    )


def _select_demos(raw_demos, cfg, index_split: str):
    split_idx = int(_train_split(cfg) * len(raw_demos))
    if index_split == "train":
        return raw_demos[:split_idx]
    if index_split == "test":
        return raw_demos[split_idx:]
    if index_split == "full":
        return raw_demos
    raise ValueError(f"Unsupported index split: {index_split}")


def build_retrieval_index(
    cfg,
    *,
    index_env_ids: list[int],
    state_dim: int,
    index_split: str,
) -> RetrievalIndex:
    states: list[np.ndarray] = []
    phases: list[float] = []
    actions: list[torch.Tensor] = []
    demo_steps: list[int] = []
    demo_count = 0

    for env_id in index_env_ids:
        data_path = (
            Path(str(cfg.dataset.root_dir))
            / str(cfg.dataset.task)
            / f"env_{env_id}"
            / "episodes_observations.pkl.gz"
        )
        raw_demos = load_pickled_data(data_path)
        demos = _select_demos(raw_demos, cfg, index_split)
        for trajectory in demos:
            if len(trajectory) < 2:
                continue
            demo_count += 1
            keypoints = _discover_keypoints_if_needed(cfg, trajectory)
            num_steps = len(trajectory) - 1
            demo_steps.append(num_steps)
            for t in range(num_steps):
                state = _low_dim_from_obs(trajectory[t], cfg)
                if state.shape[0] < state_dim:
                    raise ValueError(
                        f"state_dim={state_dim} exceeds observed low_dim_state={state.shape[0]}"
                    )
                states.append(state[:state_dim])
                phases.append(float(t) / float(max(1, num_steps - 1)))
                actions.append(_build_action(cfg, trajectory, t, keypoints=keypoints))

    if not states:
        raise ValueError(
            f"No retrieval samples loaded for task={cfg.dataset.task} env_ids={index_env_ids}"
        )

    states_tensor = torch.tensor(np.stack(states), dtype=torch.float32)
    phases_tensor = torch.tensor(phases, dtype=torch.float32).unsqueeze(1)
    actions_tensor = torch.stack(actions).to(torch.float32)
    state_mean = states_tensor.mean(dim=0, keepdim=True)
    state_std = states_tensor.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    phase_mean = phases_tensor.mean(dim=0, keepdim=True)
    phase_std = phases_tensor.std(dim=0, keepdim=True).clamp_min(1.0e-6)

    return RetrievalIndex(
        states=states_tensor,
        phases=phases_tensor,
        actions=actions_tensor,
        state_mean=state_mean,
        state_std=state_std,
        phase_mean=phase_mean,
        phase_std=phase_std,
        median_demo_steps=int(np.median(demo_steps)) if demo_steps else 1,
        demo_count=demo_count,
        sample_count=len(states),
    )


class DemoRetrievalAgent:
    def __init__(
        self,
        index: RetrievalIndex,
        *,
        state_dim: int,
        phase_weight: float,
        phase_horizon: int,
        top_k: int,
        softmax_temperature: float,
    ) -> None:
        self.index = index
        self.state_dim = int(state_dim)
        self.phase_weight = float(phase_weight)
        self.phase_horizon = max(1, int(phase_horizon))
        self.top_k = max(1, int(top_k))
        self.softmax_temperature = max(1.0e-6, float(softmax_temperature))
        self._step = 0
        self._episode_distances: list[float] = []
        self._episode_indices: list[int] = []

        self._index_state_z = (index.states - index.state_mean) / index.state_std
        self._index_phase_z = (
            ((index.phases - index.phase_mean) / index.phase_std) * self.phase_weight
        )
        self._index_features = torch.cat(
            (self._index_state_z, self._index_phase_z), dim=1
        )

    def eval(self):
        return self

    def reset_episode(self) -> None:
        self._step = 0
        self._episode_distances = []
        self._episode_indices = []

    def get_episode_metrics(self) -> dict:
        if not self._episode_distances:
            return {
                "mean_nn_distance": 0.0,
                "min_nn_distance": 0.0,
                "max_nn_distance": 0.0,
                "num_policy_queries": 0,
                "first_match_index": -1,
                "last_match_index": -1,
            }
        distances = np.asarray(self._episode_distances, dtype=np.float32)
        return {
            "mean_nn_distance": float(distances.mean()),
            "min_nn_distance": float(distances.min()),
            "max_nn_distance": float(distances.max()),
            "num_policy_queries": int(len(distances)),
            "first_match_index": int(self._episode_indices[0]),
            "last_match_index": int(self._episode_indices[-1]),
        }

    def get_action(self, front_rgb, low_dim_state, obs_context=None):  # noqa: ARG002
        query_state = low_dim_state.detach().cpu().to(torch.float32)[0, : self.state_dim]
        query_state_z = (query_state.unsqueeze(0) - self.index.state_mean) / self.index.state_std
        query_phase = min(1.0, float(self._step) / float(max(1, self.phase_horizon - 1)))
        query_phase_tensor = torch.tensor([[query_phase]], dtype=torch.float32)
        query_phase_z = (
            (query_phase_tensor - self.index.phase_mean) / self.index.phase_std
        ) * self.phase_weight
        query_feature = torch.cat((query_state_z, query_phase_z), dim=1)

        distances = torch.sum((self._index_features - query_feature) ** 2, dim=1)
        k = min(self.top_k, distances.numel())
        if k == 1:
            best_idx = int(torch.argmin(distances).item())
            action = self.index.actions[best_idx]
            best_distance = float(distances[best_idx].item())
        else:
            top_distances, top_indices = torch.topk(distances, k=k, largest=False)
            weights = torch.softmax(-top_distances / self.softmax_temperature, dim=0)
            action = torch.sum(self.index.actions[top_indices] * weights[:, None], dim=0)
            best_idx = int(top_indices[0].item())
            best_distance = float(top_distances[0].item())

        self._episode_distances.append(best_distance)
        self._episode_indices.append(best_idx)
        self._step += 1
        return action.unsqueeze(0).to(low_dim_state.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-config-path", default=None)
    parser.add_argument("--rollout-split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--env-ids", default=None)
    parser.add_argument("--index-env-ids", default=None)
    parser.add_argument("--index-split", default="train", choices=["train", "test", "full"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--phase-weight", type=float, default=0.5)
    parser.add_argument("--phase-horizon", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--softmax-temperature", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = (
        Path(args.dataset_config_path)
        if args.dataset_config_path is not None
        else BASE_DATASET_CONFIG_PATH
    )
    cfg = load_runtime_config(config_path, task=args.task)
    if bool(getattr(cfg.dataset, "include_camera_in_state", False)):
        os.environ["ELSA_INCLUDE_CAMERA_IN_STATE"] = "1"

    index_cfg = copy.deepcopy(cfg)
    index_env_ids = parse_env_ids(args.index_env_ids, _configured_env_ids(index_cfg))
    start = time.perf_counter()
    index = build_retrieval_index(
        index_cfg,
        index_env_ids=index_env_ids,
        state_dim=int(args.state_dim),
        index_split=args.index_split,
    )
    phase_horizon = int(args.phase_horizon) if args.phase_horizon > 0 else index.median_demo_steps
    agent = DemoRetrievalAgent(
        index,
        state_dim=int(args.state_dim),
        phase_weight=float(args.phase_weight),
        phase_horizon=phase_horizon,
        top_k=int(args.top_k),
        softmax_temperature=float(args.softmax_temperature),
    )

    rollout_root_dir, default_env_ids = resolve_split(cfg, args.rollout_split)
    env_ids = parse_env_ids(args.env_ids, default_env_ids)
    cfg.dataset.root_dir = rollout_root_dir
    cfg.dataset.enable_live_eval = True
    cfg.dataset.num_episodes_live = int(args.episodes)

    base_cfg = OmegaConf.load(f"{rollout_root_dir}/{args.task}/{args.task}_fed.yaml")
    base_cfg.dataset = cfg.dataset
    base_cfg.transform = cfg.transform
    transform = get_image_transform(cfg)

    per_env_rewards: dict[str, list[float]] = {}
    per_env_episode_metrics: dict[str, list[dict]] = {}
    all_rewards: list[float] = []
    for env_id in env_ids:
        task_env, rlbench_env = load_task_environment(base_cfg, env_id, headless=True)
        try:
            env_episode_metrics = []
            env_rewards = []
            for _ in range(int(args.episodes)):
                agent.reset_episode()
                episode = rollout_episode(
                    agent,
                    task_env,
                    transform,
                    torch.device("cpu"),
                    base_cfg.transform.action_min,
                    base_cfg.transform.action_max,
                    int(args.max_steps),
                    base_cfg,
                    capture_frames=False,
                )
                retrieval_metrics = agent.get_episode_metrics()
                reward = float(episode["reward"])
                env_rewards.append(reward)
                all_rewards.append(reward)
                env_episode_metrics.append(
                    {
                        "reward": reward,
                        "success": bool(episode["success"]),
                        "steps": int(episode["steps"]),
                        **retrieval_metrics,
                    }
                )
            per_env_rewards[str(env_id)] = [float(x) for x in env_rewards]
            per_env_episode_metrics[str(env_id)] = env_episode_metrics
        finally:
            rlbench_env.shutdown()

    payload = {
        "policy": "demo_state_phase_nearest_neighbor",
        "task": args.task,
        "resolved_config_path": str(config_path),
        "rollout_split": args.rollout_split,
        "env_ids": env_ids,
        "index_root_dir": str(index_cfg.dataset.root_dir),
        "index_env_ids": index_env_ids,
        "index_split": args.index_split,
        "retrieval": {
            "state_dim": int(args.state_dim),
            "phase_weight": float(args.phase_weight),
            "phase_horizon": int(phase_horizon),
            "top_k": int(args.top_k),
            "softmax_temperature": float(args.softmax_temperature),
            "demo_count": int(index.demo_count),
            "sample_count": int(index.sample_count),
            "median_demo_steps": int(index.median_demo_steps),
        },
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "episodes_per_env": int(args.episodes),
        "max_steps": int(args.max_steps),
        "sr": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_sr": float(np.std(all_rewards)) if all_rewards else 0.0,
        "num_rollouts": len(all_rewards),
        "per_env_rewards": per_env_rewards,
        "per_env_episode_metrics": per_env_episode_metrics,
        "elapsed_sec": float(time.perf_counter() - start),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
