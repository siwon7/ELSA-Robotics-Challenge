import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlbench.action_modes.action_mode import ActionMode


ACTION_PIPELINE_PRESETS = {
    "legacy_auto": {
        "action_representation": None,
        "execution_action_interface": None,
        "execution_action_adapter": None,
        "joint_velocity_servo_gain": 20.0,
        "joint_velocity_servo_clip": 1.0,
        "joint_velocity_servo_steps": 1,
        "joint_velocity_servo_tolerance": 0.01,
    },
    "joint_velocity_direct": {
        "action_representation": "joint_velocity",
        "execution_action_interface": "joint_velocity",
        "execution_action_adapter": "none",
        "joint_velocity_servo_gain": 20.0,
        "joint_velocity_servo_clip": 1.0,
        "joint_velocity_servo_steps": 1,
        "joint_velocity_servo_tolerance": 0.01,
    },
    "joint_position_to_benchmark_joint_velocity_servo": {
        "action_representation": "joint_position_absolute",
        "execution_action_interface": "joint_velocity",
        "execution_action_adapter": "joint_position_to_joint_velocity_servo",
        "joint_velocity_servo_gain": 20.0,
        "joint_velocity_servo_clip": 1.0,
        "joint_velocity_servo_steps": 2,
        "joint_velocity_servo_tolerance": 0.01,
    },
    "joint_position_direct": {
        "action_representation": "joint_position_absolute",
        "execution_action_interface": "joint_position",
        "execution_action_adapter": "none",
        "joint_velocity_servo_gain": 20.0,
        "joint_velocity_servo_clip": 1.0,
        "joint_velocity_servo_steps": 1,
        "joint_velocity_servo_tolerance": 0.01,
    },
}


def _get_rlbench_eval_classes():
    from colosseum import TASKS_PY_FOLDER, TASKS_TTM_FOLDER
    from colosseum.rlbench.extensions.environment import EnvironmentExt
    from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.action_modes.arm_action_modes import JointPosition, JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete

    class MoveArmThenGripperEval(ActionMode):
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

    class MoveArmThenGripperJointPositionEval(ActionMode):
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

    return {
        "EnvironmentExt": EnvironmentExt,
        "ObservationConfigExt": ObservationConfigExt,
        "name_to_class": name_to_class,
        "TASKS_PY_FOLDER": TASKS_PY_FOLDER,
        "TASKS_TTM_FOLDER": TASKS_TTM_FOLDER,
        "JointPosition": JointPosition,
        "JointVelocity": JointVelocity,
        "Discrete": Discrete,
        "MoveArmThenGripperEval": MoveArmThenGripperEval,
        "MoveArmThenGripperJointPositionEval": MoveArmThenGripperJointPositionEval,
    }


def get_action_representation(config) -> str:
    explicit = getattr(config.dataset, "action_representation", None)
    if explicit not in (None, "", "auto"):
        return str(explicit)
    preset = get_action_pipeline_preset(config)
    preset_value = ACTION_PIPELINE_PRESETS[preset]["action_representation"]
    if preset_value is not None:
        return str(preset_value)
    return "joint_velocity"


def get_action_pipeline_preset(config) -> str:
    explicit = getattr(config.dataset, "action_pipeline_preset", None)
    if explicit in (None, ""):
        return "legacy_auto"
    preset = str(explicit)
    if preset not in ACTION_PIPELINE_PRESETS:
        raise ValueError(
            f"Unsupported action_pipeline_preset: {preset}. "
            f"Expected one of {sorted(ACTION_PIPELINE_PRESETS)}"
        )
    return preset


def get_action_chunk_len(config) -> int:
    return int(getattr(config.dataset, "action_chunk_len", 1) or 1)


def get_receding_horizon_execute_steps(config) -> int:
    return int(getattr(config.dataset, "receding_horizon_execute_steps", 1) or 1)


def get_execution_action_interface(config) -> str:
    explicit = getattr(config.dataset, "execution_action_interface", None)
    if explicit not in (None, "", "auto"):
        return str(explicit)
    preset = get_action_pipeline_preset(config)
    preset_value = ACTION_PIPELINE_PRESETS[preset]["execution_action_interface"]
    if preset_value is not None:
        return str(preset_value)
    action_representation = get_action_representation(config)
    if action_representation.startswith("joint_position"):
        return "joint_position"
    return "joint_velocity"


def get_execution_action_adapter(config) -> str:
    explicit = getattr(config.dataset, "execution_action_adapter", None)
    if explicit not in (None, "", "auto"):
        return str(explicit)
    preset = get_action_pipeline_preset(config)
    preset_value = ACTION_PIPELINE_PRESETS[preset]["execution_action_adapter"]
    if preset_value is not None:
        return str(preset_value)
    action_representation = get_action_representation(config)
    execution_interface = get_execution_action_interface(config)
    if (
        execution_interface == "joint_velocity"
        and action_representation.startswith("joint_position")
    ):
        return "joint_position_to_joint_velocity_servo"
    return "none"


def get_joint_velocity_servo_gain(config) -> float:
    explicit = getattr(config.dataset, "joint_velocity_servo_gain", None)
    if explicit not in (None, ""):
        return float(explicit)
    preset = get_action_pipeline_preset(config)
    return float(ACTION_PIPELINE_PRESETS[preset]["joint_velocity_servo_gain"])


def get_joint_velocity_servo_clip(config) -> float:
    explicit = getattr(config.dataset, "joint_velocity_servo_clip", None)
    if explicit not in (None, ""):
        return float(explicit)
    preset = get_action_pipeline_preset(config)
    return float(ACTION_PIPELINE_PRESETS[preset]["joint_velocity_servo_clip"])


def get_joint_velocity_servo_steps(config) -> int:
    explicit = getattr(config.dataset, "joint_velocity_servo_steps", None)
    if explicit not in (None, ""):
        return int(explicit)
    preset = get_action_pipeline_preset(config)
    return int(ACTION_PIPELINE_PRESETS[preset]["joint_velocity_servo_steps"])


def get_joint_velocity_servo_tolerance(config) -> float:
    explicit = getattr(config.dataset, "joint_velocity_servo_tolerance", None)
    if explicit not in (None, ""):
        return float(explicit)
    preset = get_action_pipeline_preset(config)
    return float(ACTION_PIPELINE_PRESETS[preset]["joint_velocity_servo_tolerance"])


def get_action_output_activation(config) -> str:
    explicit = getattr(config.model, "action_output_activation", None)
    if explicit not in (None, ""):
        return str(explicit)
    action_representation = get_action_representation(config)
    if action_representation.startswith("joint_position"):
        return "identity"
    return "tanh"


def expand_action_bounds(action_min, action_max, action_dim: int):
    action_min_tensor = torch.as_tensor(action_min, dtype=torch.float32)
    action_max_tensor = torch.as_tensor(action_max, dtype=torch.float32)
    if action_min_tensor.numel() == action_dim:
        return action_min_tensor, action_max_tensor
    if action_dim % action_min_tensor.numel() != 0:
        raise ValueError(
            f"Cannot expand bounds of dim {action_min_tensor.numel()} to action_dim={action_dim}"
        )
    repeat_factor = action_dim // action_min_tensor.numel()
    return action_min_tensor.repeat(repeat_factor), action_max_tensor.repeat(repeat_factor)


def select_receding_horizon_action(action, config):
    chunk_len = get_action_chunk_len(config)
    if chunk_len <= 1:
        return action
    base_action_dim = len(config.transform.action_min)
    if action.shape[-1] < base_action_dim:
        raise ValueError(
            f"Action dim {action.shape[-1]} is smaller than base_action_dim={base_action_dim}"
        )
    return action[..., :base_action_dim]


def select_receding_horizon_actions(action, config):
    chunk_len = get_action_chunk_len(config)
    execute_steps = get_receding_horizon_execute_steps(config)
    base_action_dim = len(config.transform.action_min)
    if chunk_len <= 1:
        return [action[..., :base_action_dim]]
    if action.shape[-1] < base_action_dim:
        raise ValueError(
            f"Action dim {action.shape[-1]} is smaller than base_action_dim={base_action_dim}"
        )
    max_actions = action.shape[-1] // base_action_dim
    if max_actions <= 0:
        raise ValueError(
            f"Action dim {action.shape[-1]} cannot be reshaped into base_action_dim={base_action_dim}"
        )
    steps = min(max(1, execute_steps), chunk_len, max_actions)
    return [
        action[..., idx * base_action_dim : (idx + 1) * base_action_dim]
        for idx in range(steps)
    ]

def get_image_transform(config_file):
    transform = transforms.Compose([
        # transforms.Resize(64),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=config_file.transform.normalize_mean,
            std=config_file.transform.normalize_std
        )
    ])
    
    return transform

def process_obs(obs, transform=None):
    # invert axis from rgb to bgr
    front_image = torch.tensor(obs.front_rgb, dtype=torch.float32).permute(2, 0, 1)/255
    front_image = transform(front_image)

    # process observations for agent
    low_dim_state = torch.tensor(np.concatenate((obs.joint_positions, np.array([obs.gripper_open]))), dtype=torch.float32)

    return front_image, low_dim_state

def reverse_process_image(image):
    image = (image * 0.5) + 0.5
    image = (image.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    return image

def normalize_action(action, action_min, action_max):
    # Scale action to [0, 1] then map to [-1, 1]
    normalized_action = 2 * ((action - action_min) / (action_max - action_min)) - 1
    return normalized_action

def denormalize_action(normalized_action, action_min, action_max):
    # Map from [-1, 1] back to [0, 1] then to the original range
    normalized_action = torch.as_tensor(normalized_action, dtype=torch.float32).clamp(-1.0, 1.0)
    action = ((normalized_action + 1) / 2) * (action_max - action_min) + action_min
    return action


def joint_position_target_to_joint_velocity_action(
    target_action: np.ndarray,
    obs,
    config,
) -> np.ndarray:
    target_action = np.asarray(target_action, dtype=np.float32).reshape(-1)
    if target_action.shape[0] < 8:
        raise ValueError(
            f"Expected at least 8 dims for joint-position target, got {target_action.shape[0]}"
        )
    target_joint_positions = target_action[:7]
    current_joint_positions = np.asarray(obs.joint_positions, dtype=np.float32)
    joint_delta = target_joint_positions - current_joint_positions
    servo_gain = get_joint_velocity_servo_gain(config)
    servo_clip = get_joint_velocity_servo_clip(config)
    joint_velocity = np.clip(
        servo_gain * joint_delta,
        -servo_clip,
        servo_clip,
    ).astype(np.float32)
    gripper_action = np.asarray(target_action[7:8], dtype=np.float32)
    return np.concatenate((joint_velocity, gripper_action), axis=0)


def execute_action_with_adapter(task_env, obs, action: np.ndarray, config):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    execution_interface = get_execution_action_interface(config)
    execution_adapter = get_execution_action_adapter(config)

    if (
        execution_interface == "joint_velocity"
        and execution_adapter == "joint_position_to_joint_velocity_servo"
    ):
        servo_steps = max(1, get_joint_velocity_servo_steps(config))
        servo_tolerance = get_joint_velocity_servo_tolerance(config)
        reward = 0.0
        terminated = False
        next_obs = obs
        frames = []
        executed_steps = 0
        for _ in range(servo_steps):
            env_action = joint_position_target_to_joint_velocity_action(
                action,
                next_obs,
                config,
            )
            next_obs, step_reward, terminate = task_env.step(env_action)
            frames.append(np.asarray(next_obs.front_rgb, dtype=np.uint8))
            reward = float(step_reward)
            terminated = bool(terminate)
            executed_steps += 1
            remaining_delta = np.asarray(action[:7], dtype=np.float32) - np.asarray(
                next_obs.joint_positions,
                dtype=np.float32,
            )
            if terminated or np.max(np.abs(remaining_delta)) < servo_tolerance:
                break
        return next_obs, reward, terminated, executed_steps, frames

    next_obs, step_reward, terminate = task_env.step(action)
    return (
        next_obs,
        float(step_reward),
        bool(terminate),
        1,
        [np.asarray(next_obs.front_rgb, dtype=np.uint8)],
    )

def save_video_trajectory(front_images, video_path='/home/omniverse/Workspace/elsa_robotic_manipulation/elelsa_robotic_manipulationsa/videos', video_name='slide_block_to_target.mp4'):
        # create a video

    os.makedirs(video_path, exist_ok=True)

    video_path = os.path.join(video_path, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (128, 128))

    for img in front_images:
        img = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()

    # video to gif
    import moviepy.editor as mp
    clip = mp.VideoFileClip(video_path)
    clip.write_gif(video_path.replace('mp4', 'gif'))    

    
    print(f"Video saved at {video_path}")


def load_environment(base_cfg, collection_cfg, idx_environment, headless=True):
    """ Load the environment with the given configuration.

    Args:
        base_cfg (BaseConfig): The base configuration.
        collection_cfg (CollectionConfig): The collection configuration.
        idx_environment (int): The index of the environment to load.
        headless (bool): Whether to run the environment in headless mode.

    Returns:
        EnvironmentExt: The loaded environment.
    """
    
    rlbench_cls = _get_rlbench_eval_classes()
    ObservationConfigExt = rlbench_cls["ObservationConfigExt"]
    name_to_class = rlbench_cls["name_to_class"]
    EnvironmentExt = rlbench_cls["EnvironmentExt"]
    TASKS_PY_FOLDER = rlbench_cls["TASKS_PY_FOLDER"]
    TASKS_TTM_FOLDER = rlbench_cls["TASKS_TTM_FOLDER"]
    JointPosition = rlbench_cls["JointPosition"]
    JointVelocity = rlbench_cls["JointVelocity"]
    Discrete = rlbench_cls["Discrete"]
    MoveArmThenGripperEval = rlbench_cls["MoveArmThenGripperEval"]
    MoveArmThenGripperJointPositionEval = rlbench_cls["MoveArmThenGripperJointPositionEval"]

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

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

    obs_config = ObservationConfigExt(data_cfg)
    if os.getenv("ELSA_RECORD_GRIPPER_CLOSING", "0") == "1":
        obs_config.record_gripper_closing = True

    execution_interface = get_execution_action_interface(base_cfg)
    if execution_interface == "joint_position":
        action_mode = MoveArmThenGripperJointPositionEval(
            arm_action_mode=JointPosition(absolute_mode=True),
            gripper_action_mode=Discrete(),
        )
    elif execution_interface == "joint_velocity":
        action_mode = MoveArmThenGripperEval(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        )
    else:
        raise ValueError(f"Unsupported execution_action_interface: {execution_interface}")

    rlbench_env = EnvironmentExt(
        action_mode=action_mode,
        obs_config=obs_config,
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
