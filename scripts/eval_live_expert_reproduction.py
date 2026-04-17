import argparse
import json
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from elsa_learning_agent.utils import load_environment
from scripts.live_eval_common import load_main_split_cfg, serialize_random_state

METHOD_EXPERT_SUCCESS = "expert_success"
METHOD_STORED_JOINT_VEL = "live_demo_replay_stored_joint_vel"
METHOD_FINITE_DIFF = "live_demo_replay_finite_diff"
METHOD_TARGET_JOINT_VEL = "live_demo_replay_target_joint_vel"
DEFAULT_DT = 0.05


def parse_env_ids(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("env ids are empty")
    return values


def build_action(
    obs, next_obs, dt: float, method: str, target_joint_vel: np.ndarray | None = None
) -> np.ndarray:
    if method == METHOD_STORED_JOINT_VEL:
        joint = np.asarray(getattr(obs, "joint_velocities"), dtype=np.float32)
    elif method == METHOD_FINITE_DIFF:
        joint = (
            np.asarray(getattr(next_obs, "joint_positions"), dtype=np.float32)
            - np.asarray(getattr(obs, "joint_positions"), dtype=np.float32)
        ) / float(dt)
    elif method == METHOD_TARGET_JOINT_VEL:
        if target_joint_vel is None:
            raise ValueError("target_joint_vel is required for target joint replay")
        joint = np.asarray(target_joint_vel, dtype=np.float32)
    else:
        raise ValueError(f"unsupported action method: {method}")
    gripper = np.asarray(
        [getattr(next_obs, "gripper_open", getattr(obs, "gripper_open", 1.0))],
        dtype=np.float32,
    )
    return np.concatenate((joint, gripper), axis=0)


def sequence_from_demo(demo: Any):
    if hasattr(demo, "observations"):
        return list(getattr(demo, "observations"))
    if isinstance(demo, (list, tuple)):
        return list(demo)
    return list(demo)


def collect_live_demo_entries(task_env, amount: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for _ in range(amount):
        target_joint_vels: list[np.ndarray] = []

        def _recorder(_obs):
            target_joint_vels.append(
                np.asarray(task_env._robot.arm.get_joint_target_velocities(), dtype=np.float32)
            )

        demo = task_env.get_demos(amount=1, live_demos=True, callable_each_step=_recorder)[0]
        entries.append(
            {
                "demo": demo,
                "target_joint_velocities": target_joint_vels,
            }
        )
    return entries


def call_reset_to_demo(task_env, demo, demo_idx: int):
    if not hasattr(task_env, "reset_to_demo"):
        raise RuntimeError("task_env.reset_to_demo is not available")
    reset_to_demo = getattr(task_env, "reset_to_demo")
    last_error = None
    for fn_args in ((demo,), (demo_idx,), ()):
        try:
            return reset_to_demo(*fn_args)
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(
        "reset_to_demo did not accept expected signatures (demo,) or (demo_idx,) or ()"
    ) from last_error


def save_replay_pack(
    output_prefix: Path,
    env_id: int,
    demo_idx: int,
    demo: Any,
    dt: float,
    target_joint_velocities: list[np.ndarray] | None = None,
) -> Path:
    sequence = sequence_from_demo(demo)
    action_steps = len(sequence) - 1
    target_joint_velocities = target_joint_velocities or []
    if target_joint_velocities:
        action_steps = min(action_steps, len(target_joint_velocities))
    payload = {
        "task": output_prefix.parent.name,
        "env_id": int(env_id),
        "demo_idx": int(demo_idx),
        "random_seed": serialize_random_state(getattr(demo, "random_seed", None)),
        "joint_positions": [np.asarray(obs.joint_positions, dtype=np.float32) for obs in sequence],
        "joint_velocities": [np.asarray(obs.joint_velocities, dtype=np.float32) for obs in sequence],
        "gripper_open": [float(getattr(obs, "gripper_open", 1.0)) for obs in sequence],
        "replay_actions": {
            "stored_joint_vel": [
                build_action(sequence[t], sequence[t + 1], dt=dt, method=METHOD_STORED_JOINT_VEL).tolist()
                for t in range(action_steps)
            ],
            "finite_diff": [
                build_action(sequence[t], sequence[t + 1], dt=dt, method=METHOD_FINITE_DIFF).tolist()
                for t in range(action_steps)
            ],
            "target_joint_vel": [
                build_action(
                    sequence[t],
                    sequence[t + 1],
                    dt=dt,
                    method=METHOD_TARGET_JOINT_VEL,
                    target_joint_vel=target_joint_velocities[t],
                ).tolist()
                for t in range(action_steps)
            ] if target_joint_velocities else [],
        },
    }
    dump_path = output_prefix.parent / f"{output_prefix.stem}.env{env_id:03d}.demo{demo_idx:03d}.replay.pkl"
    with open(dump_path, "wb") as fh:
        pickle.dump(payload, fh)
    return dump_path


def run_expert_success(task_env, demo_entries: list[dict[str, Any]], env_id: int, output_prefix: Path, dt: float) -> dict[str, Any]:
    demo_results = []
    for demo_idx, entry in enumerate(demo_entries):
        demo = entry["demo"]
        dump_path = save_replay_pack(
            output_prefix,
            env_id,
            demo_idx,
            demo,
            dt,
            target_joint_velocities=entry.get("target_joint_velocities"),
        )
        demo_results.append(
            {
                "demo_idx": demo_idx,
                "success": True,
                "num_steps": int(len(sequence_from_demo(demo))),
                "has_random_seed": getattr(demo, "random_seed", None) is not None,
                "pack_path": str(dump_path),
            }
        )
    return {
        "env_id": env_id,
        "num_demos": len(demo_entries),
        "num_success": len(demo_entries),
        "sr": 1.0 if demo_entries else 0.0,
        "demo_results": demo_results,
        "method": METHOD_EXPERT_SUCCESS,
    }


def run_replay_with_reset_to_demo(
    task_env, demo_entries: list[dict[str, Any]], env_id: int, method: str, dt: float, output_prefix: Path
) -> dict[str, Any]:
    demo_results = []
    rewards = []
    success_flags = []

    for demo_idx, entry in enumerate(demo_entries):
        demo = entry["demo"]
        demo_entry: dict[str, Any] = {
            "demo_idx": demo_idx,
            "action_source": method,
            "has_random_seed": getattr(demo, "random_seed", None) is not None,
        }
        dump_path = save_replay_pack(
            output_prefix,
            env_id,
            demo_idx,
            demo,
            dt,
            target_joint_velocities=entry.get("target_joint_velocities"),
        )
        demo_entry["pack_path"] = str(dump_path)
        try:
            sequence = sequence_from_demo(demo)
            if len(sequence) < 2:
                raise ValueError("demo has fewer than 2 observations")

            call_reset_to_demo(task_env, demo, demo_idx)
            reward = 0.0
            terminated = False
            steps = 0
            target_joint_velocities = entry.get("target_joint_velocities") or []
            for t in range(len(sequence) - 1):
                target_joint_vel = None
                if method == METHOD_TARGET_JOINT_VEL:
                    if t >= len(target_joint_velocities):
                        raise ValueError(
                            f"missing target_joint_vel for demo_idx={demo_idx}, step={t}"
                        )
                    target_joint_vel = target_joint_velocities[t]
                action = build_action(
                    sequence[t],
                    sequence[t + 1],
                    dt=dt,
                    method=method,
                    target_joint_vel=target_joint_vel,
                )
                _obs, step_reward, terminate = task_env.step(action)
                reward = float(step_reward)
                steps += 1
                if terminate:
                    terminated = bool(terminate)
                    break

            success = terminated or (reward > 0.0)
            rewards.append(reward)
            success_flags.append(1.0 if success else 0.0)
            demo_entry.update(
                {
                    "reward": reward,
                    "terminated": bool(terminated),
                    "success": bool(success),
                    "steps": int(steps),
                }
            )
        except Exception as exc:
            demo_entry.update(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "reward": None,
                    "terminated": False,
                    "steps": 0,
                }
            )
            rewards.append(0.0)
            success_flags.append(0.0)
        demo_results.append(demo_entry)

    return {
        "env_id": env_id,
        "num_demos": len(demo_entries),
        "num_success": int(sum(1 for v in success_flags if v > 0.5)),
        "sr": float(np.mean(success_flags)) if success_flags else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "std_reward": float(np.std(rewards)) if rewards else None,
        "method": method,
        "demo_results": demo_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--env-ids", required=True)
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument(
        "--method",
        required=True,
        choices=[METHOD_EXPERT_SUCCESS, METHOD_STORED_JOINT_VEL, METHOD_FINITE_DIFF, METHOD_TARGET_JOINT_VEL],
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env_ids = parse_env_ids(args.env_ids)
    cfg, collection_cfg = load_main_split_cfg(args.task, args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    start = time.perf_counter()

    results = []
    all_success = []

    for env_id in env_ids:
        task_env, rlbench_env = load_environment(cfg, collection_cfg, env_id, headless=True)
        try:
            demo_entries = collect_live_demo_entries(task_env, args.num_demos)
            if args.method == METHOD_EXPERT_SUCCESS:
                env_result = run_expert_success(task_env, demo_entries, env_id, output_path, args.dt)
            else:
                env_result = run_replay_with_reset_to_demo(
                    task_env, demo_entries, env_id, args.method, args.dt, output_path
                )
            results.append(env_result)
            all_success.extend(
                [1.0 if item.get("success", False) else 0.0 for item in env_result["demo_results"]]
            )
            progress_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "task": args.task,
                        "split": args.split,
                        "method": args.method,
                        "env_ids": env_ids,
                        "completed_envs": len(results),
                        "completed_demos": len(all_success),
                        "partial_sr": float(np.mean(all_success)) if all_success else None,
                        "elapsed_sec": float(time.perf_counter() - start),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            rlbench_env.shutdown()

    payload = {
        "task": args.task,
        "split": args.split,
        "env_ids": env_ids,
        "method": args.method,
        "dt": args.dt,
        "num_demos": args.num_demos,
        "results": results,
        "sr": float(np.mean(all_success)) if all_success else 0.0,
        "elapsed_sec": float(time.perf_counter() - start),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
