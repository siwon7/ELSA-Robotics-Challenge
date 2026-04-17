import argparse
import cv2
import json
import os
import pickle
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from pyrep.const import PYREP_SCRIPT_TYPE
from pyrep.backend.simConst import sim_handle_all

from colosseum.rlbench.utils import save_demo
from elsa_learning_agent.utils import load_environment
from rlbench.demo import Demo


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


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def image_metrics(obs_a, obs_b):
    img_a = obs_a.front_rgb.astype(np.float32)
    img_b = obs_b.front_rgb.astype(np.float32)
    return {
        "reset_image_mae": float(np.abs(img_a - img_b).mean()),
        "reset_image_rmse": float(np.sqrt(((img_a - img_b) ** 2).mean())),
        "reset_equal_pixel_ratio": float((obs_a.front_rgb == obs_b.front_rgb).mean()),
    }


def serialize_random_seed_state(state):
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


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


def get_configuration_tree_bytes(scene, object_handle):
    _, _, _, buffer = scene.pyrep.script_call(
        "getConfigurationTreeBuffer@PyRep",
        PYREP_SCRIPT_TYPE,
        ints=[int(object_handle)],
    )
    return buffer


def capture_configuration_state(task_env, start_obs):
    scene = task_env._scene
    task = scene.task
    return {
        "state_kind": "configuration_tree_bytes",
        "scene_config_tree": get_configuration_tree_bytes(scene, sim_handle_all),
        "task_config_tree": get_configuration_tree_bytes(scene, task.get_base().get_handle()),
        "arm_config_tree": get_configuration_tree_bytes(scene, scene.robot.arm.get_handle()),
        "gripper_config_tree": get_configuration_tree_bytes(
            scene, scene.robot.gripper.get_handle()
        ),
        "front_camera_extrinsics": np.asarray(
            start_obs.misc["front_camera_extrinsics"], dtype=np.float64
        ),
        "variation_index": int(start_obs.misc["variation_index"]),
    }


def collect_exact_demo(task_env):
    scene = task_env._scene
    random_seed = np.random.get_state()
    descriptions, _ = task_env.reset()
    pre_record_obs = scene.get_observation()
    pre_record_state = capture_configuration_state(task_env, pre_record_obs)
    pre_record_success, pre_record_terminate = task_env._task.success()
    scene.pyrep.step()
    start_obs = scene.get_observation()
    start_state = capture_configuration_state(task_env, start_obs)
    start_success, start_terminate = task_env._task.success()

    recorded_obs = []
    executed_demo_actions = []
    recorded_configuration_states = []
    previous_obs = start_obs
    trace = {
        "physics_stepped_since_record": False,
        "pending_gripper_step": False,
        "pending_gripper_target": None,
        "pending_grasp": False,
    }

    original_pyrep_step = scene.pyrep.step
    original_actuate = scene.robot.gripper.actuate
    original_grasp = scene.robot.gripper.grasp

    def traced_pyrep_step(*args, **kwargs):
        trace["physics_stepped_since_record"] = True
        return original_pyrep_step(*args, **kwargs)

    def traced_actuate(amount, velocity):
        trace["pending_gripper_step"] = True
        trace["pending_gripper_target"] = float(amount)
        return original_actuate(amount, velocity)

    def traced_grasp(*args, **kwargs):
        trace["pending_grasp"] = True
        return original_grasp(*args, **kwargs)

    def record_step(obs):
        nonlocal previous_obs
        arm_delta = float(
            np.max(
                np.abs(
                    np.asarray(obs.joint_positions, dtype=np.float64)
                    - np.asarray(previous_obs.joint_positions, dtype=np.float64)
                )
            )
        )
        gripper_delta = float(
            np.max(
                np.abs(
                    np.asarray(obs.gripper_joint_positions, dtype=np.float64)
                    - np.asarray(previous_obs.gripper_joint_positions, dtype=np.float64)
                )
            )
        )
        if trace["pending_grasp"]:
            kind = "grasp_attach"
        elif trace["pending_gripper_step"]:
            kind = "gripper_step"
        elif trace["physics_stepped_since_record"] and arm_delta > 1e-5:
            kind = "arm_step"
        elif trace["physics_stepped_since_record"]:
            kind = "passive_step"
        else:
            kind = "observe_only"

        arm_action = None
        if kind == "arm_step":
            arm_action = np.asarray(
                scene.robot.arm.get_joint_target_positions(), dtype=np.float64
            )

        recorded_obs.append(obs)
        recorded_configuration_states.append(capture_configuration_state(task_env, obs))
        executed_demo_actions.append(
            {
                "kind": kind,
                "arm_joint_position_action": (
                    None if arm_action is None else np.asarray(arm_action, dtype=np.float64)
                ),
                "gripper_target": (
                    trace["pending_gripper_target"]
                    if trace["pending_gripper_step"]
                    else float(obs.gripper_open)
                ),
            }
        )
        previous_obs = obs
        trace["physics_stepped_since_record"] = False
        trace["pending_gripper_step"] = False
        trace["pending_gripper_target"] = None
        trace["pending_grasp"] = False

    scene.pyrep.step = traced_pyrep_step
    scene.robot.gripper.actuate = traced_actuate
    scene.robot.gripper.grasp = traced_grasp
    try:
        replay_tail = scene.get_demo(
            record=False,
            callable_each_step=record_step,
            randomly_place=False,
        )
    finally:
        scene.pyrep.step = original_pyrep_step
        scene.robot.gripper.actuate = original_actuate
        scene.robot.gripper.grasp = original_grasp
    demo = Demo(
        [start_obs] + recorded_obs,
        random_seed=random_seed,
        num_reset_attempts=getattr(replay_tail, "num_reset_attempts", None),
    )
    return {
        "descriptions": descriptions,
        "demo": demo,
        "executed_demo_actions": executed_demo_actions,
        "recorded_configuration_states": recorded_configuration_states,
        "pre_record_state": pre_record_state,
        "start_state": start_state,
        "pre_record_success": bool(pre_record_success),
        "pre_record_terminate": bool(pre_record_terminate),
        "start_success": bool(start_success),
        "start_terminate": bool(start_terminate),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, required=True)
    parser.add_argument("--split", default="eval", choices=["training", "eval", "test"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    base_cfg = OmegaConf.load("dataset_config.yaml")
    live_cfg = build_live_cfg(base_cfg, args.task, args.split)
    collection_cfg = load_collection_cfg(live_cfg.dataset.root_dir, args.task)
    task_env, rlbench_env = load_environment(
        live_cfg,
        collection_cfg,
        args.env_id,
        headless=args.headless,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    try:
        ctr_loop = task_env._robot.arm.joints[0].is_control_loop_enabled()
        task_env._robot.arm.set_control_loop_enabled(True)
        for episode_idx in range(args.num_episodes):
            user_seed = args.base_seed + episode_idx
            set_all_seeds(user_seed)
            task_env.set_variation(0)
            collection = collect_exact_demo(task_env)
            descriptions = collection["descriptions"]
            demo = collection["demo"]
            executed_demo_actions = collection["executed_demo_actions"]
            recorded_configuration_states = collection["recorded_configuration_states"]
            exact_start_state = collection["start_state"]
            pre_record_state = collection["pre_record_state"]
            final_success, final_terminate = task_env._task.success()

            demo.restore_state()
            descriptions, reset_obs = task_env.reset(demo)
            metrics = image_metrics(reset_obs, demo[0])

            episode_dir = output_dir / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            video_dir = episode_dir / "videos"
            video_paths = write_videos(
                [obs.front_rgb.copy() for obs in demo],
                video_dir / f"{args.task}_env_{args.env_id}_episode_{episode_idx}",
                fps=args.fps,
            )
            save_demo(live_cfg.data, demo, str(episode_dir), variation=0)
            with open(episode_dir / "exact_start_state.pkl", "wb") as fh:
                pickle.dump(exact_start_state, fh)
            with open(episode_dir / "pre_record_state.pkl", "wb") as fh:
                pickle.dump(pre_record_state, fh)
            with open(episode_dir / "executed_demo_actions.pkl", "wb") as fh:
                pickle.dump(executed_demo_actions, fh)
            with open(episode_dir / "recorded_configuration_states.pkl", "wb") as fh:
                pickle.dump(recorded_configuration_states, fh)

            with open(episode_dir / "variation_descriptions.pkl", "wb") as fh:
                pickle.dump(descriptions, fh)

            metadata = {
                "episode_idx": episode_idx,
                "task": args.task,
                "env_id": args.env_id,
                "split": args.split,
                "user_seed": user_seed,
                "variation_index": int(reset_obs.misc["variation_index"]),
                "num_steps": len(demo),
                "demo_random_seed": serialize_random_seed_state(demo.random_seed),
                "pre_record_success": collection["pre_record_success"],
                "pre_record_terminate": collection["pre_record_terminate"],
                "start_success": collection["start_success"],
                "start_terminate": collection["start_terminate"],
                "collection_final_success": bool(final_success),
                "collection_final_terminate": bool(final_terminate),
                "video_paths": video_paths,
                **metrics,
            }
            (episode_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            summaries.append(metadata)
            print(
                f"episode={episode_idx} user_seed={user_seed} steps={len(demo)} "
                f"reset_image_mae={metrics['reset_image_mae']:.6f} "
                f"equal_ratio={metrics['reset_equal_pixel_ratio']:.6f}"
            )
    finally:
        task_env._robot.arm.set_control_loop_enabled(ctr_loop)
        rlbench_env.shutdown()

    summary = {
        "task": args.task,
        "env_id": args.env_id,
        "split": args.split,
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "fps": args.fps,
        "headless": bool(args.headless),
        "success_rate_collection_final": (
            float(np.mean([episode["collection_final_success"] for episode in summaries]))
            if summaries
            else 0.0
        ),
        "episodes": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
