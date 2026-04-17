import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from pyrep.const import PYREP_SCRIPT_TYPE
from pyrep.const import ObjectType
from pyrep.objects.joint import Joint
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.conditions import ConditionSet, JointCondition, OrConditions
from scipy.spatial.transform import Rotation

from elsa_learning_agent.dataset.compat import load_pickled_data
from elsa_learning_agent.utils import load_environment


def build_split_root(cfg, split):
    if split == "training":
        return cfg.dataset.root_dir
    if split == "eval":
        return cfg.dataset.root_eval_dir
    if split == "test":
        return cfg.dataset.root_test_dir
    raise ValueError(f"Unsupported split: {split}")


def build_live_cfg(base_cfg, task, split):
    live_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    live_cfg.dataset.task = task
    live_cfg.dataset.root_dir = build_split_root(base_cfg, split)

    fed_cfg = OmegaConf.load(
        os.path.join(live_cfg.dataset.root_dir, task, f"{task}_fed.yaml")
    )
    live_cfg.env = fed_cfg.env
    live_cfg.data = fed_cfg.data
    return live_cfg


def load_collection_cfg(root_dir, task):
    collection_cfg_path = os.path.join(root_dir, task, f"{task}_fed.json")
    with open(collection_cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_demo_episodes(root_dir, task, env_id):
    data_path = os.path.join(root_dir, task, f"env_{env_id}", "episodes_observations.pkl.gz")
    demos = load_pickled_data(data_path)
    if hasattr(demos, "data"):
        demos = demos.data
    return data_path, demos


def load_raw_demo_episodes(raw_demo_dir):
    raw_demo_root = Path(raw_demo_dir)
    episode_dirs = sorted(
        path for path in raw_demo_root.glob("episode_*") if path.is_dir()
    )
    demos = []
    for episode_dir in episode_dirs:
        with open(episode_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)
        front_rgb_paths = sorted((episode_dir / "front_rgb").glob("*.png"))
        for obs, front_rgb_path in zip(demo, front_rgb_paths):
            if getattr(obs, "front_rgb", None) is None:
                obs.front_rgb = imageio.imread(front_rgb_path)
        exact_state_path = episode_dir / "exact_start_state.pkl"
        if exact_state_path.exists():
            with open(exact_state_path, "rb") as fh:
                demo.exact_start_state = pickle.load(fh)
        pre_record_state_path = episode_dir / "pre_record_state.pkl"
        if pre_record_state_path.exists():
            with open(pre_record_state_path, "rb") as fh:
                demo.pre_record_state = pickle.load(fh)
        executed_actions_path = episode_dir / "executed_demo_actions.pkl"
        if executed_actions_path.exists():
            with open(executed_actions_path, "rb") as fh:
                demo.executed_demo_actions = pickle.load(fh)
        recorded_states_path = episode_dir / "recorded_configuration_states.pkl"
        if recorded_states_path.exists():
            with open(recorded_states_path, "rb") as fh:
                demo.recorded_configuration_states = pickle.load(fh)
        demos.append(demo)
    return str(raw_demo_root), demos


def gt_action_for_step(current_obs, next_obs):
    return np.concatenate(
        (current_obs.joint_velocities, np.array([next_obs.gripper_open], dtype=np.float64))
    ).astype(np.float32)


def write_mp4_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_avi_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_gif_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(round(1000.0 / float(fps))), 1)
    imageio.mimsave(str(output_path), frames, format="GIF", duration=duration_ms, loop=0)


def write_videos(frames, base_path, fps):
    mp4_path = base_path.with_suffix(".mp4")
    avi_path = base_path.with_suffix(".avi")
    gif_path = base_path.with_suffix(".gif")
    write_mp4_video(frames, mp4_path, fps)
    write_avi_video(frames, avi_path, fps)
    write_gif_video(frames, gif_path, fps)
    return {
        "mp4": str(mp4_path),
        "avi": str(avi_path),
        "gif": str(gif_path),
    }


def pose_to_matrix(pose):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose[3:], dtype=np.float64)).as_matrix()
    matrix[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return matrix


def restore_task_low_dim_state(task_env, first_demo_obs):
    task = task_env._task
    state = np.asarray(first_demo_obs.task_low_dim_state, dtype=np.float64).reshape(-1)
    if not hasattr(task, "_initial_objs_in_scene") or not task._initial_objs_in_scene:
        task.set_initial_objects_in_scene()

    cursor = 0
    anchor_obj = None
    anchor_pose = None
    for obj, _ in task._initial_objs_in_scene:
        if obj.get_parent() == task.get_base():
            anchor_obj = obj
            anchor_pose = state[cursor : cursor + 7]
            break
        cursor += 7

    if anchor_obj is not None:
        anchor_rel_matrix = anchor_obj.get_matrix(relative_to=task.get_base())
        target_anchor_matrix = pose_to_matrix(anchor_pose)
        target_base_matrix = target_anchor_matrix @ np.linalg.inv(anchor_rel_matrix)
        task.get_base().set_matrix(target_base_matrix)

    cursor = 0
    for obj, objtype in task._initial_objs_in_scene:
        cursor += 7

        if objtype == ObjectType.JOINT:
            Joint(obj.get_handle()).set_joint_position(float(state[cursor]))
            cursor += 1
        elif objtype == ObjectType.FORCE_SENSOR:
            cursor += 6

    if cursor != len(state):
        raise ValueError(
            f"Task low-dim state length mismatch: consumed={cursor}, total={len(state)}"
        )

    if "front_camera_extrinsics" in first_demo_obs.misc:
        VisionSensor("cam_front").set_matrix(
            np.asarray(first_demo_obs.misc["front_camera_extrinsics"], dtype=np.float64)
        )

    task_env._scene.pyrep.step()
    task.step()
    return get_observation_safe(task_env._scene)


def restore_snapshot_state(task_env, snapshot):
    scene = task_env._scene
    scene.reset()
    task_env._reset_called = True
    robot = scene.robot
    task = scene.task
    if snapshot.get("state_kind") == "configuration_tree_bytes":
        if "scene_config_tree" in snapshot:
            scene.pyrep.script_call(
                "setConfigurationTreeBuffer@PyRep",
                PYREP_SCRIPT_TYPE,
                bytes=snapshot["scene_config_tree"],
            )
        else:
            for buffer in [
                snapshot["task_config_tree"],
                snapshot["arm_config_tree"],
                snapshot["gripper_config_tree"],
            ]:
                scene.pyrep.script_call(
                    "setConfigurationTreeBuffer@PyRep",
                    PYREP_SCRIPT_TYPE,
                    bytes=buffer,
                )
    elif snapshot.get("state_kind") == "configuration_tree":
        task.restore_state(snapshot["task_state"])
        scene.pyrep.set_configuration_tree(snapshot["arm_config_tree"])
        scene.pyrep.set_configuration_tree(snapshot["gripper_config_tree"])
    else:
        robot.arm.set_joint_positions(
            np.asarray(snapshot["arm_joint_positions"], dtype=np.float64).tolist(),
            disable_dynamics=True,
        )
        robot.arm.set_joint_target_velocities(
            [0.0] * len(snapshot["arm_joint_positions"])
        )
        robot.gripper.set_joint_positions(
            np.asarray(snapshot["gripper_joint_positions"], dtype=np.float64).tolist(),
            disable_dynamics=True,
        )
        robot.gripper.set_joint_target_velocities(
            [0.0] * len(snapshot["gripper_joint_positions"])
        )

        base = task.get_base()
        object_lookup = {
            obj.get_name(): obj for obj in base.get_objects_in_tree(exclude_base=False)
        }
        for obj_state in snapshot["object_states"]:
            obj = object_lookup[obj_state["name"]]
            if obj_state["name"] == snapshot["base_name"]:
                obj.set_matrix(np.asarray(obj_state["matrix"], dtype=np.float64))
            elif obj_state["type"] == ObjectType.JOINT.name:
                Joint(obj.get_handle()).set_joint_position(float(obj_state["joint_position"]))
            else:
                obj.set_matrix(
                    np.asarray(obj_state["matrix"], dtype=np.float64), relative_to=base
                )
    if "front_camera_extrinsics" in snapshot:
        VisionSensor("cam_front").set_matrix(
            np.asarray(snapshot["front_camera_extrinsics"], dtype=np.float64)
        )
    return get_observation_safe(scene)


def refresh_joint_condition_baselines(task):
    def recurse(condition):
        if isinstance(condition, JointCondition):
            condition._original_pos = condition._joint.get_joint_position()
        elif isinstance(condition, (ConditionSet, OrConditions)):
            for child in condition._conditions:
                recurse(child)

    for condition in task._success_conditions:
        recurse(condition)
    for condition in task._fail_conditions:
        recurse(condition)


def compute_image_metrics(obs, demo_obs):
    current = obs.front_rgb.astype(np.float32)
    target = demo_obs.front_rgb.astype(np.float32)
    return {
        "initial_image_mae": float(np.abs(current - target).mean()),
        "initial_image_rmse": float(np.sqrt(((current - target) ** 2).mean())),
        "initial_equal_pixel_ratio": float((obs.front_rgb == demo_obs.front_rgb).mean()),
    }


def get_observation_safe(scene):
    try:
        return scene.get_observation()
    except RuntimeError as exc:
        if "No value available yet." not in str(exc):
            raise
        previous_joint_forces = scene._obs_config.joint_forces
        scene._obs_config.joint_forces = False
        try:
            return scene.get_observation()
        finally:
            scene._obs_config.joint_forces = previous_joint_forces


def step_with_joint_state(task_env, target_obs):
    scene = task_env._scene
    robot = scene.robot
    robot.arm.set_joint_positions(target_obs.joint_positions.tolist())
    robot.arm.set_joint_target_positions(target_obs.joint_positions.tolist())
    robot.arm.set_joint_target_velocities([0.0] * len(target_obs.joint_positions))
    robot.gripper.set_joint_positions(target_obs.gripper_joint_positions.tolist())
    robot.gripper.set_joint_target_velocities([0.0] * len(target_obs.gripper_joint_positions))
    scene.pyrep.step()
    scene.task.step()
    obs = get_observation_safe(scene)
    success, terminate = task_env._task.success()
    reward = 1.0 if success else 0.0
    return obs, reward, bool(success or terminate)


def step_with_executed_demo_action(task_env, action_record):
    scene = task_env._scene
    robot = scene.robot
    gripper = robot.gripper
    kind = action_record["kind"]
    arm_action = action_record.get("arm_joint_position_action")
    gripper_target = action_record.get("gripper_target")

    if kind == "arm_step":
        robot.arm.set_joint_target_positions(np.asarray(arm_action, dtype=np.float64).tolist())
        scene.step()
    elif kind == "gripper_step":
        target = float(gripper_target)
        if target > 0.5:
            gripper.release()
        gripper.actuate(target, 0.04)
        scene.pyrep.step()
        scene.task.step()
    elif kind == "grasp_attach":
        for graspable in scene.task.get_graspable_objects():
            gripper.grasp(graspable)
    elif kind == "passive_step":
        scene.pyrep.step()
        scene.task.step()
    elif kind == "observe_only":
        pass
    else:
        raise ValueError(f"Unsupported executed demo action kind: {kind}")

    obs = get_observation_safe(scene)
    success, terminate = task_env._task.success()
    reward = 1.0 if success else 0.0
    return obs, reward, bool(success or terminate)


def step_with_recorded_configuration_state(task_env, state_record):
    obs = restore_snapshot_state(task_env, state_record)
    success, terminate = task_env._task.success()
    reward = 1.0 if success else 0.0
    return obs, reward, bool(success or terminate)


def replay_episode(
    task_env,
    demo,
    max_steps,
    reset_mode="restore_state",
    replay_mode="velocity",
    continue_after_success=False,
):
    initial_frames = []
    if reset_mode == "demo_reset":
        if hasattr(demo, "restore_state"):
            demo.restore_state()
        _, obs = task_env.reset(demo)
    elif reset_mode == "exact_state":
        if not hasattr(demo, "exact_start_state"):
            raise ValueError("Demo does not contain an exact_start_state snapshot.")
        obs = restore_snapshot_state(task_env, demo.exact_start_state)
    elif reset_mode == "pre_record_state":
        if not hasattr(demo, "pre_record_state"):
            raise ValueError("Demo does not contain a pre_record_state snapshot.")
        obs = restore_snapshot_state(task_env, demo.pre_record_state)
        initial_frames.append(obs.front_rgb.copy())
        task_env._scene.pyrep.step()
        task_env._scene.task.step()
        obs = task_env._scene.get_observation()
    elif reset_mode == "restore_state":
        _, obs = task_env.reset()
        obs = restore_task_low_dim_state(task_env, demo[0])
    elif reset_mode == "plain_reset":
        _, obs = task_env.reset()
    else:
        raise ValueError(f"Unsupported reset mode: {reset_mode}")
    initial_metrics = compute_image_metrics(obs, demo[0])

    frames = initial_frames + [obs.front_rgb.copy()]
    initial_success, initial_terminate = task_env._task.success()
    reward = 0.0
    max_reward = 0.0
    terminate = bool(initial_terminate)
    terminated_seen = bool(initial_terminate)
    executed_steps = 0
    first_success_step = 0 if initial_success else None
    if initial_success:
        max_reward = 1.0

    for time_step in range(min(len(demo) - 1, max_steps)):
        if terminate and not continue_after_success:
            break
        if replay_mode == "joint_state":
            obs, reward, terminate = step_with_joint_state(task_env, demo[time_step + 1])
        elif replay_mode == "executed_demo":
            if not hasattr(demo, "executed_demo_actions"):
                raise ValueError("Demo does not contain executed_demo_actions.")
            if len(demo.executed_demo_actions) != len(demo) - 1:
                raise ValueError(
                    "executed_demo_actions length does not match demo transitions: "
                    f"{len(demo.executed_demo_actions)} vs {len(demo) - 1}"
                )
            obs, reward, terminate = step_with_executed_demo_action(
                task_env, demo.executed_demo_actions[time_step]
            )
        elif replay_mode == "state_sequence":
            if not hasattr(demo, "recorded_configuration_states"):
                raise ValueError("Demo does not contain recorded_configuration_states.")
            if len(demo.recorded_configuration_states) != len(demo) - 1:
                raise ValueError(
                    "recorded_configuration_states length does not match demo transitions: "
                    f"{len(demo.recorded_configuration_states)} vs {len(demo) - 1}"
                )
            obs, reward, terminate = step_with_recorded_configuration_state(
                task_env, demo.recorded_configuration_states[time_step]
            )
        elif replay_mode == "velocity":
            action = gt_action_for_step(demo[time_step], demo[time_step + 1])
            obs, reward, terminate = task_env.step(action)
        else:
            raise ValueError(f"Unsupported replay mode: {replay_mode}")
        if reward > max_reward:
            max_reward = float(reward)
            if first_success_step is None and reward > 0.0:
                first_success_step = int(time_step + 1)
        terminated_seen = terminated_seen or bool(terminate)
        frames.append(obs.front_rgb.copy())
        executed_steps += 1

    result = {
        "reward": float(reward),
        "max_reward": float(max_reward),
        "success_ever": bool(max_reward > 0.0),
        "first_success_step": first_success_step,
        "terminate": bool(terminated_seen),
        "executed_steps": int(executed_steps),
        "demo_steps": int(len(demo) - 1),
    }
    result.update(initial_metrics)
    return result, frames


def parse_episode_ids(total_episodes, episode_ids_arg, num_episodes):
    if episode_ids_arg:
        episode_ids = [int(token.strip()) for token in episode_ids_arg.split(",") if token.strip()]
    else:
        episode_ids = list(range(min(num_episodes, total_episodes)))

    invalid = [episode_id for episode_id in episode_ids if episode_id < 0 or episode_id >= total_episodes]
    if invalid:
        raise ValueError(f"Episode ids out of range: {invalid}")
    return episode_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, required=True)
    parser.add_argument(
        "--split",
        default="eval",
        choices=["training", "eval", "test"],
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-ids", default=None)
    parser.add_argument("--save-video-count", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--raw-demo-dir", default=None)
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument(
        "--replay-mode",
        default="velocity",
        choices=["velocity", "joint_state", "executed_demo", "state_sequence"],
    )
    parser.add_argument(
        "--reset-mode",
        default=None,
        choices=["demo_reset", "exact_state", "pre_record_state", "restore_state", "plain_reset"],
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_cfg = OmegaConf.load("dataset_config.yaml")
    live_cfg = build_live_cfg(base_cfg, args.task, args.split)
    collection_cfg = load_collection_cfg(live_cfg.dataset.root_dir, args.task)
    if args.raw_demo_dir:
        data_path, demos = load_raw_demo_episodes(args.raw_demo_dir)
    else:
        data_path, demos = load_demo_episodes(live_cfg.dataset.root_dir, args.task, args.env_id)
    episode_ids = parse_episode_ids(len(demos), args.episode_ids, args.num_episodes)
    reset_mode = args.reset_mode or ("demo_reset" if args.raw_demo_dir else "restore_state")

    task_env, rlbench_env = load_environment(
        live_cfg,
        collection_cfg,
        args.env_id,
        headless=args.headless,
    )

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    episode_results = []
    rewards = []
    restore_control_loop = None

    try:
        if args.replay_mode in {"executed_demo", "state_sequence"}:
            restore_control_loop = task_env._robot.arm.joints[0].is_control_loop_enabled()
            task_env._robot.arm.set_control_loop_enabled(True)
        task_env.reset()
        for position, episode_id in enumerate(episode_ids):
            result, frames = replay_episode(
                task_env=task_env,
                demo=demos[episode_id],
                max_steps=args.max_steps,
                reset_mode=reset_mode,
                replay_mode=args.replay_mode,
                continue_after_success=args.continue_after_success,
            )
            result["episode_id"] = int(episode_id)
            if position < args.save_video_count:
                video_base = video_dir / f"{args.task}_env_{args.env_id}_episode_{episode_id}"
                result["video_paths"] = write_videos(frames, video_base, fps=args.fps)
            episode_results.append(result)
            rewards.append(result["reward"])
            print(
                f"episode={episode_id} reward={result['reward']:.4f} "
                f"terminate={result['terminate']} "
                f"steps={result['executed_steps']}/{result['demo_steps']}"
            )
    finally:
        if restore_control_loop is not None:
            task_env._robot.arm.set_control_loop_enabled(restore_control_loop)
        rlbench_env.shutdown()

    success_flags = [reward > 0.0 for reward in rewards]
    success_ever_flags = [episode["success_ever"] for episode in episode_results]
    summary = {
        "task": args.task,
        "split": args.split,
        "env_id": args.env_id,
        "data_path": data_path,
        "episode_ids": episode_ids,
        "num_episodes": len(episode_ids),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "success_rate_reward_gt0": float(np.mean(success_flags)) if success_flags else 0.0,
        "success_rate_any_reward_gt0": (
            float(np.mean(success_ever_flags)) if success_ever_flags else 0.0
        ),
        "max_steps": args.max_steps,
        "headless": bool(args.headless),
        "continue_after_success": bool(args.continue_after_success),
        "reset_mode": reset_mode,
        "replay_mode": args.replay_mode,
        "episodes": episode_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
