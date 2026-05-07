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
from elsa_learning_agent.live_rollout import load_task_environment, rollout_episode
from elsa_learning_agent.utils import (
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_image_transform,
)
from scripts.eval_demo_retrieval_policy_live import (
    _build_action,
    _configured_env_ids,
    _discover_keypoints_if_needed,
    _low_dim_from_obs,
    _select_demos,
    parse_env_ids,
    resolve_split,
)

try:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


@dataclass(frozen=True)
class DemoTrajectory:
    demo_id: int
    env_id: int
    states: torch.Tensor
    actions: torch.Tensor


@dataclass(frozen=True)
class TrajectoryLibrary:
    trajectories: list[DemoTrajectory]
    initial_states: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    demo_count: int
    sample_count: int
    median_demo_steps: int


def build_trajectory_library(
    cfg,
    *,
    index_env_ids: list[int],
    state_dim: int,
    index_split: str,
) -> TrajectoryLibrary:
    trajectories: list[DemoTrajectory] = []
    all_states: list[torch.Tensor] = []
    demo_steps: list[int] = []
    demo_id = 0

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
            keypoints = _discover_keypoints_if_needed(cfg, trajectory)
            states = []
            actions = []
            num_steps = len(trajectory) - 1
            for t in range(num_steps):
                state = _low_dim_from_obs(trajectory[t], cfg)
                if state.shape[0] < state_dim:
                    raise ValueError(
                        f"state_dim={state_dim} exceeds observed low_dim_state={state.shape[0]}"
                    )
                states.append(torch.tensor(state[:state_dim], dtype=torch.float32))
                actions.append(_build_action(cfg, trajectory, t, keypoints=keypoints))
            if not states:
                continue
            state_tensor = torch.stack(states)
            action_tensor = torch.stack(actions).to(torch.float32)
            trajectories.append(
                DemoTrajectory(
                    demo_id=demo_id,
                    env_id=int(env_id),
                    states=state_tensor,
                    actions=action_tensor,
                )
            )
            all_states.append(state_tensor)
            demo_steps.append(num_steps)
            demo_id += 1

    if not trajectories:
        raise ValueError(
            f"No trajectory samples loaded for task={cfg.dataset.task} env_ids={index_env_ids}"
        )

    state_matrix = torch.cat(all_states, dim=0)
    state_mean = state_matrix.mean(dim=0, keepdim=True)
    state_std = state_matrix.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    initial_states = torch.stack([traj.states[0] for traj in trajectories])

    return TrajectoryLibrary(
        trajectories=trajectories,
        initial_states=initial_states,
        state_mean=state_mean,
        state_std=state_std,
        demo_count=len(trajectories),
        sample_count=int(sum(len(traj.actions) for traj in trajectories)),
        median_demo_steps=int(np.median(demo_steps)) if demo_steps else 1,
    )


class DemoTrajectoryAgent:
    def __init__(
        self,
        library: TrajectoryLibrary,
        *,
        state_dim: int,
        alignment: str,
        local_window: int,
        phase_weight: float,
    ) -> None:
        if alignment not in {"initial_replay", "local_state", "global_state"}:
            raise ValueError(f"Unsupported alignment: {alignment}")
        self.library = library
        self.state_dim = int(state_dim)
        self.alignment = alignment
        self.local_window = max(0, int(local_window))
        self.phase_weight = float(phase_weight)
        self._selected: DemoTrajectory | None = None
        self._step = 0
        self._selected_demo_id = -1
        self._initial_distance = 0.0
        self._match_distances: list[float] = []
        self._match_indices: list[int] = []

        self._initial_z = self._normalize(library.initial_states)
        self._all_states_z = [
            self._normalize(traj.states) for traj in self.library.trajectories
        ]

    def _normalize(self, states: torch.Tensor) -> torch.Tensor:
        return (states.to(torch.float32) - self.library.state_mean) / self.library.state_std

    def eval(self):
        return self

    def reset_episode(self) -> None:
        self._selected = None
        self._step = 0
        self._selected_demo_id = -1
        self._initial_distance = 0.0
        self._match_distances = []
        self._match_indices = []

    def _select_trajectory(self, query_state: torch.Tensor) -> None:
        query_z = self._normalize(query_state.unsqueeze(0))
        distances = torch.sum((self._initial_z - query_z) ** 2, dim=1)
        idx = int(torch.argmin(distances).item())
        self._selected = self.library.trajectories[idx]
        self._selected_demo_id = int(self._selected.demo_id)
        self._initial_distance = float(distances[idx].item())

    def _select_action_index(self, query_state: torch.Tensor) -> tuple[int, float]:
        if self._selected is None:
            raise RuntimeError("trajectory is not selected")
        if self.alignment == "initial_replay":
            idx = min(self._step, len(self._selected.actions) - 1)
            return idx, 0.0

        query_z = self._normalize(query_state.unsqueeze(0))
        if self.alignment == "local_state":
            center = min(self._step, len(self._selected.actions) - 1)
            start = max(0, center - self.local_window)
            end = min(len(self._selected.actions), center + self.local_window + 1)
            states_z = self._all_states_z[self._selected.demo_id][start:end]
            distances = torch.sum((states_z - query_z) ** 2, dim=1)
            if self.phase_weight > 0:
                phase = torch.arange(start, end, dtype=torch.float32) / float(
                    max(1, len(self._selected.actions) - 1)
                )
                query_phase = float(self._step) / float(max(1, len(self._selected.actions) - 1))
                distances = distances + self.phase_weight * (phase - query_phase) ** 2
            rel_idx = int(torch.argmin(distances).item())
            return start + rel_idx, float(distances[rel_idx].item())

        best_demo = self._selected
        best_idx = min(self._step, len(best_demo.actions) - 1)
        best_distance = float("inf")
        for traj, states_z in zip(self.library.trajectories, self._all_states_z):
            distances = torch.sum((states_z - query_z) ** 2, dim=1)
            if self.phase_weight > 0:
                phase = torch.arange(len(traj.actions), dtype=torch.float32) / float(
                    max(1, len(traj.actions) - 1)
                )
                query_phase = float(self._step) / float(max(1, len(traj.actions) - 1))
                distances = distances + self.phase_weight * (phase - query_phase) ** 2
            idx = int(torch.argmin(distances).item())
            distance = float(distances[idx].item())
            if distance < best_distance:
                best_demo = traj
                best_idx = idx
                best_distance = distance
        self._selected = best_demo
        self._selected_demo_id = int(best_demo.demo_id)
        return best_idx, best_distance

    def get_episode_metrics(self) -> dict:
        distances = np.asarray(self._match_distances, dtype=np.float32)
        return {
            "selected_demo_id": int(self._selected_demo_id),
            "initial_demo_distance": float(self._initial_distance),
            "mean_match_distance": float(distances.mean()) if distances.size else 0.0,
            "max_match_distance": float(distances.max()) if distances.size else 0.0,
            "first_match_index": int(self._match_indices[0]) if self._match_indices else -1,
            "last_match_index": int(self._match_indices[-1]) if self._match_indices else -1,
            "num_policy_queries": int(len(self._match_indices)),
        }

    def get_action(self, front_rgb, low_dim_state, obs_context=None):  # noqa: ARG002
        query_state = low_dim_state.detach().cpu().to(torch.float32)[0, : self.state_dim]
        if self._selected is None:
            self._select_trajectory(query_state)
        action_idx, distance = self._select_action_index(query_state)
        action = self._selected.actions[action_idx]
        self._match_indices.append(int(action_idx))
        self._match_distances.append(float(distance))
        self._step = max(self._step + 1, action_idx + 1 if self.alignment != "initial_replay" else self._step + 1)
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
    parser.add_argument("--top-k", type=int, default=1)  # Accepted for launcher compatibility.
    parser.add_argument(
        "--alignment",
        default="local_state",
        choices=["initial_replay", "local_state", "global_state"],
    )
    parser.add_argument("--local-window", type=int, default=8)
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
    library = build_trajectory_library(
        index_cfg,
        index_env_ids=index_env_ids,
        state_dim=int(args.state_dim),
        index_split=args.index_split,
    )
    agent = DemoTrajectoryAgent(
        library,
        state_dim=int(args.state_dim),
        alignment=str(args.alignment),
        local_window=int(args.local_window),
        phase_weight=float(args.phase_weight),
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
                trajectory_metrics = agent.get_episode_metrics()
                reward = float(episode["reward"])
                env_rewards.append(reward)
                all_rewards.append(reward)
                env_episode_metrics.append(
                    {
                        "reward": reward,
                        "success": bool(episode["success"]),
                        "steps": int(episode["steps"]),
                        **trajectory_metrics,
                    }
                )
            per_env_rewards[str(env_id)] = [float(x) for x in env_rewards]
            per_env_episode_metrics[str(env_id)] = env_episode_metrics
        finally:
            rlbench_env.shutdown()

    payload = {
        "policy": "demo_trajectory_follower",
        "task": args.task,
        "resolved_config_path": str(config_path),
        "rollout_split": args.rollout_split,
        "env_ids": env_ids,
        "index_root_dir": str(index_cfg.dataset.root_dir),
        "index_env_ids": index_env_ids,
        "index_split": args.index_split,
        "trajectory": {
            "alignment": str(args.alignment),
            "local_window": int(args.local_window),
            "state_dim": int(args.state_dim),
            "phase_weight": float(args.phase_weight),
            "demo_count": int(library.demo_count),
            "sample_count": int(library.sample_count),
            "median_demo_steps": int(library.median_demo_steps),
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
