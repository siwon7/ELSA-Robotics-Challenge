import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

from elsa_learning_agent.utils import execute_action_with_adapter, load_environment
from scripts.live_eval_common import deserialize_random_state, load_main_split_cfg


def save_gif(path: Path, frames, fps: int):
    if imageio is not None:
        imageio.mimsave(path, frames, fps=fps)
        return
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def configure_action_pipeline(cfg, method: str, args):
    if method == "joint_velocity_direct":
        cfg.dataset.action_pipeline_preset = "joint_velocity_direct"
        cfg.dataset.action_representation = "joint_velocity"
        cfg.dataset.execution_action_interface = "joint_velocity"
        cfg.dataset.execution_action_adapter = "none"
    elif method == "joint_position_direct":
        cfg.dataset.action_pipeline_preset = "joint_position_direct"
        cfg.dataset.action_representation = "joint_position_absolute"
        cfg.dataset.execution_action_interface = "joint_position"
        cfg.dataset.execution_action_adapter = "none"
    elif method == "joint_position_to_benchmark_joint_velocity_servo":
        cfg.dataset.action_pipeline_preset = "joint_position_to_benchmark_joint_velocity_servo"
        cfg.dataset.action_representation = "joint_position_absolute"
        cfg.dataset.execution_action_interface = "joint_velocity"
        cfg.dataset.execution_action_adapter = "joint_position_to_joint_velocity_servo"
        cfg.dataset.joint_velocity_servo_gain = float(args.servo_gain)
        cfg.dataset.joint_velocity_servo_clip = float(args.servo_clip)
        cfg.dataset.joint_velocity_servo_steps = int(args.servo_steps)
        cfg.dataset.joint_velocity_servo_tolerance = float(args.servo_tolerance)
    else:
        raise ValueError(f"Unsupported method: {method}")


def select_pack(pack_dir: Path, env_id: int | None, demo_idx: int | None):
    for pack_path in sorted(pack_dir.glob("*.replay.pkl")):
        with open(pack_path, "rb") as fh:
            pack = pickle.load(fh)
        if env_id is not None and int(pack["env_id"]) != int(env_id):
            continue
        if demo_idx is not None and int(pack["demo_idx"]) != int(demo_idx):
            continue
        return pack_path, pack
    raise FileNotFoundError(
        f"no replay pack matched env_id={env_id}, demo_idx={demo_idx} in {pack_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "joint_velocity_direct",
            "joint_position_direct",
            "joint_position_to_benchmark_joint_velocity_servo",
        ],
    )
    parser.add_argument("--env-id", type=int, default=None)
    parser.add_argument("--demo-idx", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-gif", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--servo-gain", type=float, default=20.0)
    parser.add_argument("--servo-clip", type=float, default=1.0)
    parser.add_argument("--servo-steps", type=int, default=2)
    parser.add_argument("--servo-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    pack_path, pack = select_pack(Path(args.pack_dir), args.env_id, args.demo_idx)
    cfg, collection_cfg = load_main_split_cfg(args.task, args.split)
    configure_action_pipeline(cfg, args.method, args)

    random_state = deserialize_random_state(pack.get("random_seed"))
    if random_state is None:
        raise ValueError("pack does not include random_seed")
    np.random.set_state(random_state)

    env_id = int(pack["env_id"])
    task_env, rlbench_env = load_environment(cfg, collection_cfg, env_id, headless=True)
    try:
        _descriptions, obs = task_env.reset()
        frames = [np.asarray(obs.front_rgb, dtype=np.uint8)]
        reward = 0.0
        terminated = False
        steps = 0

        if args.method == "joint_velocity_direct":
            actions = pack["replay_actions"]["stored_joint_vel"]
            for action in actions:
                obs, step_reward, terminate = task_env.step(np.asarray(action, dtype=np.float32))
                frames.append(np.asarray(obs.front_rgb, dtype=np.uint8))
                reward = float(step_reward)
                steps += 1
                if terminate:
                    terminated = True
                    break
        else:
            seq_positions = pack["joint_positions"]
            seq_gripper = pack["gripper_open"]
            for t in range(len(seq_positions) - 1):
                next_pos = np.asarray(seq_positions[t + 1], dtype=np.float32)
                action = np.concatenate(
                    (next_pos, np.asarray([seq_gripper[t + 1]], dtype=np.float32)),
                    axis=0,
                )
                obs, step_reward, terminate, executed_steps, step_frames = execute_action_with_adapter(
                    task_env,
                    obs,
                    action,
                    cfg,
                )
                frames.extend(step_frames)
                reward = float(step_reward)
                steps += int(executed_steps)
                if terminate:
                    terminated = True
                    break

        success = bool(terminated or reward > 0.0)
        gif_path = Path(args.output_gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        save_gif(gif_path, frames, fps=args.fps)

        payload = {
            "task": args.task,
            "split": args.split,
            "method": args.method,
            "env_id": env_id,
            "demo_idx": int(pack["demo_idx"]),
            "pack_path": str(pack_path),
            "success": success,
            "terminated": terminated,
            "reward": reward,
            "steps": steps,
            "output_gif": str(gif_path),
            "servo_gain": float(args.servo_gain),
            "servo_clip": float(args.servo_clip),
            "servo_steps": int(args.servo_steps),
            "servo_tolerance": float(args.servo_tolerance),
        }
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        rlbench_env.shutdown()


if __name__ == "__main__":
    main()
