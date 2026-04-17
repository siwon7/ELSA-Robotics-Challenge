import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import OmegaConf
from elsa_learning_agent.dataset.compat import (
    canonicalize_pose_quaternion,
    clip_action_target,
    get_action_space,
    rotvec_to_quaternion_xyzw,
)
from elsa_learning_agent.kinematics import build_low_dim_state

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

def get_image_transform(config_file):
    transform = transforms.Compose([
        transforms.Normalize(
            mean=config_file.transform.normalize_mean,
            std=config_file.transform.normalize_std
        )
    ])

    return transform

def process_obs(obs, transform=None):
    front_image = torch.tensor(obs.front_rgb, dtype=torch.float32).permute(2, 0, 1)/255
    front_image = transform(front_image)

    low_dim_state = torch.tensor(
        build_low_dim_state(obs.joint_positions, obs.gripper_open),
        dtype=torch.float32,
    )

    return front_image, low_dim_state

def reverse_process_image(image):
    image = (image * 0.5) + 0.5
    image = (image.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    return image

def normalize_action(action, action_min, action_max):
    normalized_action = 2 * ((action - action_min) / (action_max - action_min)) - 1
    return normalized_action

def denormalize_action(normalized_action, action_min, action_max):
    action = ((normalized_action + 1) / 2) * (action_max - action_min) + action_min
    return action


def get_ik_solver_name():
    return os.environ.get("ELSA_IK_SOLVER", "jacobian").strip().lower()


def get_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def prepare_action_for_env(action, config):
    action = clip_action_target(action, config)
    action = np.asarray(action, dtype=np.float32).copy()
    action_space = get_action_space(config)
    if action_space in {"ee_pose", "delta_ee_pose"}:
        action[:7] = canonicalize_pose_quaternion(action[:7])
    elif action_space == "ee_pose_rotvec":
        quat = rotvec_to_quaternion_xyzw(action[3:6])
        action = np.concatenate((action[:3], quat, action[6:7])).astype(np.float32)
    elif action_space == "delta_ee_pose_rotvec":
        quat = rotvec_to_quaternion_xyzw(action[3:6])
        action = np.concatenate((action[:3], quat, action[6:7])).astype(np.float32)
    return action


def build_arm_action_mode(action_space, ik_solver=None):
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, JointVelocity
    from elsa_learning_agent.ik_action_modes import EndEffectorPoseViaIKSampling

    solver = (ik_solver or get_ik_solver_name()).strip().lower()
    sampling_collision_checking = get_bool_env("ELSA_SAMPLING_IK_COLLISION_CHECKING", False)
    sampling_trials = int(os.environ.get("ELSA_SAMPLING_IK_TRIALS", "300"))
    sampling_max_configs = max(1, int(os.environ.get("ELSA_SAMPLING_IK_MAX_CONFIGS", "8")))
    sampling_distance_threshold = float(
        os.environ.get("ELSA_SAMPLING_IK_DISTANCE_THRESHOLD", "0.65")
    )
    sampling_max_time_ms = int(os.environ.get("ELSA_SAMPLING_IK_MAX_TIME_MS", "10"))
    if action_space == "joint_velocity":
        return JointVelocity()
    if action_space in {"ee_pose", "ee_pose_rotvec"}:
        if solver == "sampling":
            return EndEffectorPoseViaIKSampling(
                absolute_mode=True,
                frame="world",
                collision_checking=sampling_collision_checking,
                trials=sampling_trials,
                max_configs=sampling_max_configs,
                distance_threshold=sampling_distance_threshold,
                max_time_ms=sampling_max_time_ms,
            )
        if solver == "planning":
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
            return EndEffectorPoseViaPlanning(
                absolute_mode=True,
                frame="world",
                collision_checking=False,
            )
        return EndEffectorPoseViaIK(
            absolute_mode=True,
            frame="world",
            collision_checking=False,
        )
    if action_space in {"delta_ee_pose", "delta_ee_pose_rotvec"}:
        if solver == "sampling":
            return EndEffectorPoseViaIKSampling(
                absolute_mode=False,
                frame="world",
                collision_checking=sampling_collision_checking,
                trials=sampling_trials,
                max_configs=sampling_max_configs,
                distance_threshold=sampling_distance_threshold,
                max_time_ms=sampling_max_time_ms,
            )
        if solver == "planning":
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
            return EndEffectorPoseViaPlanning(
                absolute_mode=False,
                frame="world",
                collision_checking=False,
            )
        return EndEffectorPoseViaIK(
            absolute_mode=False,
            frame="world",
            collision_checking=False,
        )
    raise ValueError(f"Unsupported action_space: {action_space}")

def save_video_trajectory(front_images, video_path='/home/omniverse/Workspace/elsa_robotic_manipulation/elelsa_robotic_manipulationsa/videos', video_name='slide_block_to_target.mp4'):
    os.makedirs(video_path, exist_ok=True)

    video_path = os.path.join(video_path, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (128, 128))

    for img in front_images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()

    import moviepy.editor as mp
    clip = mp.VideoFileClip(video_path)
    clip.write_gif(video_path.replace('mp4', 'gif'))

    print(f"Video saved at {video_path}")


def load_environment(base_cfg, collection_cfg, idx_environment, headless=True):
    from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
    from colosseum.rlbench.extensions.environment import EnvironmentExt
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.gripper_action_modes import Discrete
    from colosseum import TASKS_PY_FOLDER, TASKS_TTM_FOLDER
    task = name_to_class(base_cfg.env.task_name, TASKS_PY_FOLDER)
    config = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    env_entries = collection_cfg.get("env_config", [])
    env_entry = next(
        (entry for entry in env_entries if entry.get("env_idx") == idx_environment),
        None,
    )
    if env_entry is None:
        raise ValueError(f"Environment index {idx_environment} not found in collection config")

    env_factors = config.env.scene.factors
    for variation_cfg in env_entry.get("variations_parameters", []):
        var_type = variation_cfg["type"]
        var_name = variation_cfg.get("name")
        for factor_cfg in env_factors:
            if factor_cfg.variation != var_type:
                continue
            if var_name is not None and "name" in factor_cfg and factor_cfg.name != var_name:
                continue
            for key, value in variation_cfg.items():
                if key == "type":
                    continue
                factor_cfg[key] = value
            break

    data_cfg, env_cfg = config.data, config.env
    action_space = get_action_space(config)
    arm_action_mode = build_arm_action_mode(action_space)

    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=arm_action_mode, gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_cfg),
        headless=headless,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=env_cfg,
        )

    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)

    return task_env, rlbench_env

def load_config():
    config_path = os.path.join("./dataset_config.yaml")
    config = OmegaConf.load(config_path)
    return config
