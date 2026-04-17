import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from elsa_learning_agent.utils import (
    process_obs,
    normalize_action,
    get_image_transform,
    get_action_representation,
)
from elsa_learning_agent.dataset.compat import load_pickled_data
from elsa_learning_agent.dataset.keypoint_discovery import (
    discover_heuristic_keypoints,
    find_next_keypoint_index,
)


class ImitationDataset(Dataset):
    def __init__(self, config, train=False, test=False, normalize=False):
        self.root_dir = config.dataset.root_dir
        task = config.dataset.task
        env_id = config.dataset.env_id
        train_split = float(
            getattr(
                config.dataset,
                "train_split",
                1.0 - float(getattr(config.dataset, "test_split", 0.1)),
            )
        )
        self._action_representation = get_action_representation(config)
        self._action_chunk_len = int(getattr(config.dataset, "action_chunk_len", 1) or 1)
        self._action_keyframe_horizon = int(
            getattr(config.dataset, "action_keyframe_horizon", 1) or 1
        )
        self._action_keyframe_selection = str(
            getattr(config.dataset, "action_keyframe_selection", "fixed_horizon")
        )
        self._action_keyframe_stopping_delta = float(
            getattr(config.dataset, "action_keyframe_stopping_delta", 0.1)
        )
        self._action_keyframe_stopped_buffer_steps = int(
            getattr(config.dataset, "action_keyframe_stopped_buffer_steps", 4) or 4
        )
        data_path = os.path.join(self.root_dir, f"{task}", f"env_{env_id}", "episodes_observations.pkl.gz")
        
        demos_raw_data = load_pickled_data(data_path)
        self.normalize = normalize
        self.action_min = torch.tensor(config.transform.action_min)
        self.action_max = torch.tensor(config.transform.action_max)

        split_index = int(train_split * len(demos_raw_data))
        if train:
            demos_raw_data = demos_raw_data[:split_index]
        elif test:
            demos_raw_data = demos_raw_data[split_index:]

        self.transform = get_image_transform(config)
        self.data = []
        self.demos_idx = []

        # Load data
        print("Loading dataset from:", data_path)
        for i, demo in enumerate(tqdm(demos_raw_data)):
            self.demos_idx.append(len(self.data))
            keypoints = None
            if self._action_representation == "joint_position_keyframe":
                keypoints = self._discover_keypoints(demo)
            num_steps = len(demo) - 1
            for t in range(num_steps):
                self.data.append(self._load_datapoint(demo, t, keypoints=keypoints))

    def _discover_keypoints(self, trajectory):
        if self._action_keyframe_selection == "fixed_horizon":
            return None
        if self._action_keyframe_selection == "peract_heuristic":
            return discover_heuristic_keypoints(
                trajectory,
                stopping_delta=self._action_keyframe_stopping_delta,
                stopped_buffer_steps=self._action_keyframe_stopped_buffer_steps,
            )
        raise ValueError(
            f"Unsupported action_keyframe_selection: {self._action_keyframe_selection}"
        )

    def _get_keyframe_target_index(self, trajectory, time_step, keypoints):
        if self._action_keyframe_selection == "fixed_horizon":
            return min(
                time_step + self._action_keyframe_horizon,
                len(trajectory) - 1,
            )
        if self._action_keyframe_selection == "peract_heuristic":
            return find_next_keypoint_index(
                keypoints=keypoints or [],
                time_step=time_step,
                fallback_last_index=len(trajectory) - 1,
            )
        raise ValueError(
            f"Unsupported action_keyframe_selection: {self._action_keyframe_selection}"
        )

    def _build_single_action(self, trajectory, time_step, keypoints=None):
        clamped_step = min(time_step, len(trajectory) - 2)
        obs = trajectory[clamped_step]
        next_obs = trajectory[clamped_step + 1]
        if self._action_representation == "joint_position_absolute":
            arm_action = np.asarray(next_obs.joint_positions, dtype=np.float32)
        elif self._action_representation == "joint_position_keyframe":
            target_index = self._get_keyframe_target_index(
                trajectory,
                time_step=clamped_step,
                keypoints=keypoints,
            )
            target_obs = trajectory[target_index]
            arm_action = np.asarray(target_obs.joint_positions, dtype=np.float32)
            return np.concatenate(
                (
                    arm_action,
                    np.array([target_obs.gripper_open], dtype=np.float32),
                ),
                axis=0,
            )
        elif self._action_representation == "joint_velocity":
            arm_action = np.asarray(obs.joint_velocities, dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported action_representation: {self._action_representation}"
            )
        return np.concatenate(
            (arm_action, np.array([next_obs.gripper_open], dtype=np.float32)),
            axis=0,
        )

    def _load_datapoint(self, trajectory, time_step, keypoints=None):
        obs = trajectory[time_step]
        front_image, low_dim_state = process_obs(obs, self.transform)
        action_seq = [
            self._build_single_action(
                trajectory,
                time_step + offset,
                keypoints=keypoints,
            )
            for offset in range(self._action_chunk_len)
        ]
        action = torch.tensor(np.concatenate(action_seq, axis=0), dtype=torch.float32)
        if self.normalize:
            if action.numel() % self.action_min.numel() != 0:
                raise ValueError(
                    f"Cannot normalize action of dim {action.numel()} with bounds "
                    f"of dim {self.action_min.numel()}"
                )
            repeat_factor = action.numel() // self.action_min.numel()
            action_min = self.action_min.repeat(repeat_factor)
            action_max = self.action_max.repeat(repeat_factor)
            action = normalize_action(action, action_min, action_max)
        return {
            "action": action,
            "low_dim_state": low_dim_state,
            "image": front_image,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
