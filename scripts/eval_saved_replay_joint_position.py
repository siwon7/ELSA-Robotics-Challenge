import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from elsa_learning_agent.utils import execute_action_with_adapter, load_environment
from scripts.live_eval_common import deserialize_random_state, load_main_split_cfg


def load_replay_environment(task_name: str, split: str, env_id: int, args):
    cfg, collection_cfg = load_main_split_cfg(task_name, split)
    cfg.dataset.action_representation = "joint_position_absolute"
    if args.benchmark_joint_velocity_servo:
        cfg.dataset.execution_action_interface = "joint_velocity"
        cfg.dataset.execution_action_adapter = "joint_position_to_joint_velocity_servo"
        cfg.dataset.joint_velocity_servo_gain = float(args.servo_gain)
        cfg.dataset.joint_velocity_servo_clip = float(args.servo_clip)
        cfg.dataset.joint_velocity_servo_steps = int(args.servo_steps)
        cfg.dataset.joint_velocity_servo_tolerance = float(args.servo_tolerance)
    else:
        cfg.dataset.execution_action_interface = "joint_position"
        cfg.dataset.execution_action_adapter = "none"
    task_env, rlbench_env = load_environment(cfg, collection_cfg, env_id, headless=True)
    return cfg, rlbench_env, task_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument(
        "--mode",
        default="absolute",
        choices=["absolute", "delta", "interp2", "interp3"],
    )
    parser.add_argument("--benchmark-joint-velocity-servo", action="store_true")
    parser.add_argument("--servo-gain", type=float, default=20.0)
    parser.add_argument("--servo-clip", type=float, default=1.0)
    parser.add_argument("--servo-steps", type=int, default=1)
    parser.add_argument("--servo-tolerance", type=float, default=0.01)
    parser.add_argument("--hold-steps", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pack_paths = sorted(Path(args.pack_dir).glob("*.replay.pkl"))
    if not pack_paths:
        raise FileNotFoundError(f"no replay packs found in {args.pack_dir}")

    start = time.perf_counter()
    success_flags = []
    results = []
    by_env = {}
    for pack_path in pack_paths:
        with open(pack_path, "rb") as fh:
            pack = pickle.load(fh)
        by_env.setdefault(int(pack["env_id"]), []).append((pack_path, pack))

    for env_id in sorted(by_env):
        cfg, rlbench_env, task_env = load_replay_environment(
            args.task, args.split, env_id, args
        )
        try:
            for pack_path, pack in by_env[env_id]:
                random_state = deserialize_random_state(pack.get("random_seed"))
                if random_state is None:
                    raise ValueError("pack does not include random_seed")
                np.random.set_state(random_state)
                _descriptions, obs = task_env.reset()
                reward = 0.0
                terminated = False
                seq_positions = pack["joint_positions"]
                seq_gripper = pack["gripper_open"]
                steps = 0
                for t in range(len(seq_positions) - 1):
                    current_pos = np.asarray(seq_positions[t], dtype=np.float32)
                    next_pos = np.asarray(seq_positions[t + 1], dtype=np.float32)
                    if args.mode == "absolute":
                        arm_targets = [next_pos]
                    elif args.mode == "delta":
                        arm_targets = [next_pos - current_pos]
                    elif args.mode == "interp2":
                        midpoint = (current_pos + next_pos) / 2.0
                        arm_targets = [midpoint, next_pos]
                    elif args.mode == "interp3":
                        point1 = current_pos + (next_pos - current_pos) / 3.0
                        point2 = current_pos + 2.0 * (next_pos - current_pos) / 3.0
                        arm_targets = [point1, point2, next_pos]
                    else:
                        raise ValueError(f"Unsupported mode: {args.mode}")

                    for arm in arm_targets:
                        action = np.concatenate(
                            (
                                arm,
                                np.asarray([seq_gripper[t + 1]], dtype=np.float32),
                            ),
                            axis=0,
                        )
                        for _ in range(max(1, int(args.hold_steps))):
                            obs, step_reward, terminate, executed_steps, _frames = execute_action_with_adapter(
                                task_env,
                                obs,
                                action,
                                cfg,
                            )
                            reward = float(step_reward)
                            steps += int(executed_steps)
                            if terminate:
                                terminated = True
                                break
                        if terminated:
                            break
                    if terminated:
                        break
                success = terminated or (reward > 0.0)
                success_flags.append(1.0 if success else 0.0)
                results.append(
                    {
                        "env_id": env_id,
                        "demo_idx": int(pack["demo_idx"]),
                        "reward": reward,
                        "terminated": terminated,
                        "success": success,
                        "num_actions": len(seq_positions) - 1,
                        "steps": steps,
                        "pack_path": str(pack_path),
                    }
                )
        finally:
            rlbench_env.shutdown()

    payload = {
        "task": args.task,
        "split": args.split,
        "method": f"joint_position_{args.mode}",
        "benchmark_joint_velocity_servo": bool(args.benchmark_joint_velocity_servo),
        "servo_gain": float(args.servo_gain),
        "servo_clip": float(args.servo_clip),
        "servo_steps": int(args.servo_steps),
        "servo_tolerance": float(args.servo_tolerance),
        "hold_steps": int(args.hold_steps),
        "num_packs": len(pack_paths),
        "num_success": int(sum(1 for x in success_flags if x > 0.5)),
        "sr": float(np.mean(success_flags)) if success_flags else 0.0,
        "elapsed_sec": float(time.perf_counter() - start),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
