import gzip
import os
import pickle

import numpy as np
import torch
from pyquaternion import Quaternion
from torchvision import transforms
from elsa_learning_agent.kinematics import build_low_dim_state

DEFAULT_ACTION_SPACE = "joint_velocity"
POSE_ACTION_SPACES = {
    "ee_pose",
    "ee_pose_rotvec",
    "delta_ee_pose",
    "delta_ee_pose_rotvec",
}
DEFAULT_ACTION_DIMS = {
    "joint_velocity": 8,
    "ee_pose": 8,
    "ee_pose_rotvec": 7,
    "delta_ee_pose": 8,
    "delta_ee_pose_rotvec": 7,
}


class CompatObservation:
    """Lightweight stand-in for RLBench observations used during unpickling."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class CompatDataContainer:
    """Compatibility wrapper matching the pickled dataset container layout."""

    def __init__(self):
        self.data = None

    def __setstate__(self, state):
        self.__dict__.update(state)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "colosseum.rlbench.datacontainer" and name == "DataContainer":
            return CompatDataContainer
        if module == "rlbench.backend.observation" and name == "Observation":
            return CompatObservation
        return super().find_class(module, name)


def load_pickled_data(path):
    with gzip.open(path, "rb") as file_obj:
        container = CompatUnpickler(file_obj).load()

    return container.data if hasattr(container, "data") else container


def get_image_transform(config_file):
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=config_file.transform.normalize_mean,
                std=config_file.transform.normalize_std,
            )
        ]
    )


def process_obs(obs, transform=None):
    front_image = torch.tensor(obs.front_rgb, dtype=torch.float32).permute(2, 0, 1) / 255
    if transform is not None:
        front_image = transform(front_image)

    low_dim_state = torch.tensor(
        build_low_dim_state(obs.joint_positions, obs.gripper_open),
        dtype=torch.float32,
    )
    return front_image, low_dim_state


def get_action_space(config):
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        return DEFAULT_ACTION_SPACE
    return str(model_cfg.get("action_space", DEFAULT_ACTION_SPACE)).strip().lower()


def get_action_dim(config):
    transform_cfg = getattr(config, "transform", None)
    action_min = getattr(transform_cfg, "action_min", None) if transform_cfg is not None else None
    if action_min is not None:
        return int(len(action_min))
    return DEFAULT_ACTION_DIMS[get_action_space(config)]


def is_pose_action_space(action_space):
    return action_space in POSE_ACTION_SPACES


def canonicalize_quaternion_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0:
        quat = quat / quat_norm
    if quat[-1] < 0:
        quat = -quat
    return quat


def canonicalize_pose_quaternion(pose):
    pose = np.asarray(pose, dtype=np.float32).copy()
    pose[3:7] = canonicalize_quaternion_xyzw(pose[3:7])
    return pose


def quaternion_xyzw_to_rotvec(quat):
    quat = canonicalize_quaternion_xyzw(quat)
    quat_obj = Quaternion(quat[3], quat[0], quat[1], quat[2])
    angle = float(quat_obj.angle)
    if np.isclose(angle, 0.0):
        return np.zeros(3, dtype=np.float32)
    axis = np.asarray(quat_obj.axis, dtype=np.float32)
    return (axis * angle).astype(np.float32)


def rotvec_to_quaternion_xyzw(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float32)
    angle = float(np.linalg.norm(rotvec))
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = rotvec / angle
    quat_obj = Quaternion(axis=axis, angle=angle)
    return canonicalize_quaternion_xyzw(
        np.array([quat_obj.x, quat_obj.y, quat_obj.z, quat_obj.w], dtype=np.float32)
    )


def build_delta_pose_target(obs, next_obs):
    current_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    next_pose = np.asarray(next_obs.gripper_pose, dtype=np.float32)

    delta_position = next_pose[:3] - current_pose[:3]
    current_quat = Quaternion(
        current_pose[6], current_pose[3], current_pose[4], current_pose[5]
    )
    next_quat = Quaternion(next_pose[6], next_pose[3], next_pose[4], next_pose[5])
    delta_quat = next_quat * current_quat.inverse
    delta_pose = np.concatenate(
        (
            delta_position.astype(np.float32),
            np.array(
                [delta_quat.x, delta_quat.y, delta_quat.z, delta_quat.w],
                dtype=np.float32,
            ),
        )
    )
    return canonicalize_pose_quaternion(delta_pose)


def build_delta_pose_rotvec_target(obs, next_obs):
    delta_pose = build_delta_pose_target(obs, next_obs)
    rotvec = quaternion_xyzw_to_rotvec(delta_pose[3:7])
    return np.concatenate((delta_pose[:3], rotvec)).astype(np.float32)


def build_absolute_pose_rotvec_target(next_obs):
    pose = canonicalize_pose_quaternion(next_obs.gripper_pose)
    rotvec = quaternion_xyzw_to_rotvec(pose[3:7])
    return np.concatenate((pose[:3], rotvec)).astype(np.float32)


def _get_optional_float_env(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def clip_action_target(action, config):
    action = np.asarray(action, dtype=np.float32).copy()
    action_space = get_action_space(config)
    if action_space != "delta_ee_pose_rotvec":
        return action

    pos_clip = _get_optional_float_env("ELSA_DELTA_POS_CLIP")
    rot_clip = _get_optional_float_env("ELSA_DELTA_ROT_CLIP")
    gripper_min = _get_optional_float_env("ELSA_GRIPPER_MIN_CLIP")
    gripper_max = _get_optional_float_env("ELSA_GRIPPER_MAX_CLIP")

    if pos_clip is not None:
        action[:3] = np.clip(action[:3], -pos_clip, pos_clip)
    if rot_clip is not None:
        action[3:6] = np.clip(action[3:6], -rot_clip, rot_clip)
    if gripper_min is not None or gripper_max is not None:
        lo = 0.0 if gripper_min is None else gripper_min
        hi = 1.0 if gripper_max is None else gripper_max
        action[6] = np.clip(action[6], lo, hi)
    return action


def build_action_target(obs, next_obs, config):
    action_space = get_action_space(config)
    if action_space == "joint_velocity":
        return np.concatenate(
            (obs.joint_velocities, np.array([next_obs.gripper_open], dtype=np.float32))
        ).astype(np.float32)
    if action_space == "ee_pose":
        return np.concatenate(
            (
                canonicalize_pose_quaternion(next_obs.gripper_pose),
                np.array([next_obs.gripper_open], dtype=np.float32),
            )
        ).astype(np.float32)
    if action_space == "ee_pose_rotvec":
        return np.concatenate(
            (
                build_absolute_pose_rotvec_target(next_obs),
                np.array([next_obs.gripper_open], dtype=np.float32),
            )
        ).astype(np.float32)
    if action_space == "delta_ee_pose":
        return np.concatenate(
            (
                build_delta_pose_target(obs, next_obs),
                np.array([next_obs.gripper_open], dtype=np.float32),
            )
        ).astype(np.float32)
    if action_space == "delta_ee_pose_rotvec":
        return clip_action_target(
            np.concatenate(
            (
                build_delta_pose_rotvec_target(obs, next_obs),
                np.array([next_obs.gripper_open], dtype=np.float32),
            )
        ).astype(np.float32),
            config,
        )
    raise ValueError(f"Unsupported action_space: {action_space}")


def normalize_action(action, action_min, action_max):
    return 2 * ((action - action_min) / (action_max - action_min)) - 1
