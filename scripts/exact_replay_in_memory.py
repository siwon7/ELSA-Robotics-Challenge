import argparse
import json
import random
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import collect_raw_demos as crd
import replay_ground_truth as rg
from elsa_learning_agent.utils import load_environment
from rlbench.demo import Demo


def capture_runtime_configuration_state(task_env, obs):
    return crd.capture_configuration_state(task_env, obs)


def collect_runtime_demo(task_env):
    scene = task_env._scene
    random_seed = np.random.get_state()
    descriptions, _ = task_env.reset()
    pre_record_obs = scene.get_observation()
    pre_record_state = capture_runtime_configuration_state(task_env, pre_record_obs)
    pre_record_success, pre_record_terminate = task_env._task.success()
    scene.pyrep.step()
    start_obs = scene.get_observation()
    start_state = capture_runtime_configuration_state(task_env, start_obs)
    start_success, start_terminate = task_env._task.success()

    recorded_obs = []
    recorded_configuration_states = []
    executed_demo_actions = []
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
        recorded_configuration_states.append(capture_runtime_configuration_state(task_env, obs))
        executed_demo_actions.append(
            {
                "kind": kind,
                "arm_joint_position_action": arm_action,
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
    demo.executed_demo_actions = executed_demo_actions
    demo.recorded_configuration_states = recorded_configuration_states
    demo.pre_record_state = pre_record_state
    demo.exact_start_state = start_state
    return {
        "descriptions": descriptions,
        "demo": demo,
        "pre_record_success": bool(pre_record_success),
        "pre_record_terminate": bool(pre_record_terminate),
        "start_success": bool(start_success),
        "start_terminate": bool(start_terminate),
    }


def evaluate_replay_mode(task_env, demos, task, env_id, replay_mode, output_dir, fps):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    episode_results = []
    rewards = []

    restore_control_loop = task_env._robot.arm.joints[0].is_control_loop_enabled()
    task_env._robot.arm.set_control_loop_enabled(True)
    try:
        task_env.reset()
        for episode_id, demo in enumerate(demos):
            result, frames = rg.replay_episode(
                task_env=task_env,
                demo=demo,
                max_steps=300,
                reset_mode="pre_record_state",
                replay_mode=replay_mode,
                continue_after_success=True,
            )
            video_base = video_dir / f"{task}_env_{env_id}_episode_{episode_id}"
            result["video_paths"] = rg.write_videos(frames, video_base, fps=fps)
            result["episode_id"] = episode_id
            episode_results.append(result)
            rewards.append(result["reward"])
            print(
                f"{replay_mode}: episode={episode_id} reward={result['reward']:.4f} "
                f"steps={result['executed_steps']}/{result['demo_steps']}"
            )
    finally:
        task_env._robot.arm.set_control_loop_enabled(restore_control_loop)

    summary = {
        "task": task,
        "env_id": env_id,
        "replay_mode": replay_mode,
        "num_episodes": len(demos),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "success_rate_reward_gt0": float(np.mean([r > 0.0 for r in rewards])) if rewards else 0.0,
        "success_rate_any_reward_gt0": (
            float(np.mean([episode["success_ever"] for episode in episode_results]))
            if episode_results
            else 0.0
        ),
        "episodes": episode_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, required=True)
    parser.add_argument("--split", default="eval", choices=["training", "eval", "test"])
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    base_cfg = OmegaConf.load("dataset_config.yaml")
    live_cfg = crd.build_live_cfg(base_cfg, args.task, args.split)
    collection_cfg = crd.load_collection_cfg(live_cfg.dataset.root_dir, args.task)
    task_env, rlbench_env = load_environment(
        live_cfg,
        collection_cfg,
        args.env_id,
        headless=args.headless,
    )

    demos = []
    collection_meta = []
    try:
        control_loop_enabled = task_env._robot.arm.joints[0].is_control_loop_enabled()
        task_env._robot.arm.set_control_loop_enabled(True)
        try:
            for episode_idx in range(args.num_episodes):
                seed = args.base_seed + episode_idx
                random.seed(seed)
                np.random.seed(seed)
                task_env.set_variation(0)
                collection = collect_runtime_demo(task_env)
                demo = collection["demo"]
                demos.append(demo)
                collection_meta.append(
                    {
                        "episode_id": episode_idx,
                        "user_seed": seed,
                        "num_steps": len(demo) - 1,
                        "pre_record_success": collection["pre_record_success"],
                        "start_success": collection["start_success"],
                    }
                )
                print(
                    f"collected episode={episode_idx} seed={seed} steps={len(demo) - 1}"
                )
        finally:
            task_env._robot.arm.set_control_loop_enabled(control_loop_enabled)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "collection.json").write_text(
            json.dumps({"episodes": collection_meta}, indent=2), encoding="utf-8"
        )

        evaluate_replay_mode(
            task_env, demos, args.task, args.env_id, "executed_demo", output_dir / "executed_demo", args.fps
        )
        evaluate_replay_mode(
            task_env, demos, args.task, args.env_id, "state_sequence", output_dir / "state_sequence", args.fps
        )
    finally:
        rlbench_env.shutdown()


if __name__ == "__main__":
    main()
