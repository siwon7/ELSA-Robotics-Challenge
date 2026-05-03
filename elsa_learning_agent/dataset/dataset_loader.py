import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from elsa_learning_agent.utils import (
    process_obs,
    process_obs_with_context,
    normalize_action,
    get_image_transform,
    get_action_representation,
    requires_observation_context,
    requires_ee_position,
    uses_temporal_rgb_pair,
)
from elsa_learning_agent.dataset.compat import load_pickled_data
from elsa_learning_agent.dataset.keypoint_discovery import (
    discover_heuristic_keypoints,
    find_next_keypoint_index,
)


class ImitationDataset(Dataset):
    def __init__(self, config, train=False, test=False, normalize=False):
        self.config = config
        self.root_dir = config.dataset.root_dir
        task = config.dataset.task
        env_ids_cfg = getattr(config.dataset, "env_ids", None)
        if env_ids_cfg:
            env_ids = [int(env_id) for env_id in env_ids_cfg]
        else:
            env_ids = [int(config.dataset.env_id)]
        self.env_ids = env_ids
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
        self.normalize = normalize
        self.action_min = torch.tensor(config.transform.action_min)
        self.action_max = torch.tensor(config.transform.action_max)

        self.transform = get_image_transform(config)
        self._include_obs_context = requires_observation_context(config)
        # Camera-aware baseline: append flattened K (9) + T (16) to low_dim_state.
        # Total state size becomes 8 (proprio) + 25 = 33 when enabled.
        self._include_camera_in_state = bool(
            getattr(config.dataset, "include_camera_in_state", False)
        )
        self._use_temporal_rgb_pair = uses_temporal_rgb_pair(config)
        # When EE-mask aux supervision is enabled, also load obs.gripper_pose[:3]
        # (the 3-D end-effector position) as 'ee_position' for each datapoint.
        self._include_ee_position = requires_ee_position(config)
        self.data = []
        self.demos_idx = []

        # Load data from one or more environments using the same per-env split.
        for env_id in self.env_ids:
            data_path = os.path.join(
                self.root_dir,
                f"{task}",
                f"env_{env_id}",
                "episodes_observations.pkl.gz",
            )
            demos_raw_data = load_pickled_data(data_path)

            split_index = int(train_split * len(demos_raw_data))
            if train:
                demos_raw_data = demos_raw_data[:split_index]
            elif test:
                demos_raw_data = demos_raw_data[split_index:]

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
        prev_obs = trajectory[max(0, time_step - 1)] if self._use_temporal_rgb_pair else None
        if self._include_obs_context:
            front_image, low_dim_state, obs_context = process_obs_with_context(
                obs,
                self.transform,
                prev_obs=prev_obs,
                temporal_rgb_pair=self._use_temporal_rgb_pair,
            )
        else:
            front_image, low_dim_state = process_obs(
                obs,
                self.transform,
                prev_obs=prev_obs,
                temporal_rgb_pair=self._use_temporal_rgb_pair,
            )
            obs_context = None
        if self._include_camera_in_state:
            misc = getattr(obs, "misc", {}) or {}
            K = misc.get("front_camera_intrinsics")
            T = misc.get("front_camera_extrinsics")
            if K is None or T is None:
                raise ValueError(
                    "include_camera_in_state=True but obs.misc lacks "
                    "front_camera_intrinsics or front_camera_extrinsics"
                )
            cam_state = torch.tensor(
                np.concatenate(
                    [
                        np.asarray(K, dtype=np.float32).reshape(-1),
                        np.asarray(T, dtype=np.float32).reshape(-1),
                    ]
                ),
                dtype=torch.float32,
            )
            low_dim_state = torch.cat([low_dim_state, cam_state], dim=0)
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
        datapoint = {
            "action": action,
            "low_dim_state": low_dim_state,
            "image": front_image,
        }
        if obs_context is not None:
            datapoint["obs_context"] = obs_context
        if self._include_ee_position:
            gripper_pose = getattr(obs, "gripper_pose", None)
            if gripper_pose is None:
                raise ValueError(
                    "ee_aux_loss_weight > 0 but obs.gripper_pose is missing on this RLBench observation."
                )
            ee_xyz = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)[:3]
            datapoint["ee_position"] = torch.tensor(ee_xyz, dtype=torch.float32)
        return datapoint

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
