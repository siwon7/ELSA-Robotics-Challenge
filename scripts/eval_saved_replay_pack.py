import argparse
import json
import pickle
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from elsa_learning_agent.utils import load_environment
from scripts.live_eval_common import deserialize_random_state, load_main_split_cfg


def replay_one_pack(task_env, pack: dict, method: str, hold_steps: int, control_loop: bool) -> dict:
    random_state = deserialize_random_state(pack.get("random_seed"))
    if random_state is None:
        raise ValueError("pack does not include random_seed")
    np.random.set_state(random_state)
    task_env.reset()
    arm = task_env._robot.arm
    prev_control_loop = arm.joints[0].is_control_loop_enabled()
    arm.set_control_loop_enabled(bool(control_loop))

    actions = pack["replay_actions"][method]
    reward = 0.0
    terminated = False
    try:
        for step_idx, action in enumerate(actions):
            action_array = np.asarray(action, dtype=np.float32)
            for _ in range(max(1, hold_steps)):
                _obs, step_reward, terminate = task_env.step(action_array)
                reward = float(step_reward)
                if terminate:
                    terminated = True
                    break
            if terminated:
                break
    finally:
        arm.set_control_loop_enabled(prev_control_loop)

    success = terminated or (reward > 0.0)
    return {
        "env_id": int(pack["env_id"]),
        "demo_idx": int(pack["demo_idx"]),
        "method": method,
        "reward": reward,
        "terminated": bool(terminated),
        "success": bool(success),
        "num_actions": len(actions),
        "hold_steps": int(hold_steps),
        "control_loop": bool(control_loop),
        "has_random_seed": pack.get("random_seed") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["stored_joint_vel", "finite_diff", "target_joint_vel"],
    )
    parser.add_argument("--hold-steps", type=int, default=1)
    parser.add_argument("--control-loop", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)
    pack_paths = sorted(pack_dir.glob("*.replay.pkl"))
    if not pack_paths:
        raise FileNotFoundError(f"no replay packs found in {pack_dir}")

    cfg, collection_cfg = load_main_split_cfg(args.task, args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")

    start = time.perf_counter()
    grouped = defaultdict(list)
    for pack_path in pack_paths:
        with open(pack_path, "rb") as fh:
            pack = pickle.load(fh)
        grouped[int(pack["env_id"])].append((pack_path, pack))

    results = []
    success_flags = []
    for env_id in sorted(grouped):
        task_env, rlbench_env = load_environment(cfg, collection_cfg, env_id, headless=True)
        try:
            for pack_path, pack in grouped[env_id]:
                try:
                    item = replay_one_pack(
                        task_env, pack, args.method, args.hold_steps, args.control_loop
                    )
                    item["pack_path"] = str(pack_path)
                except Exception as exc:
                    item = {
                        "env_id": int(env_id),
                        "demo_idx": int(pack.get("demo_idx", -1)),
                        "method": args.method,
                        "success": False,
                        "reward": None,
                        "terminated": False,
                        "num_actions": 0,
                        "hold_steps": int(args.hold_steps),
                        "control_loop": bool(args.control_loop),
                        "pack_path": str(pack_path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                results.append(item)
                success_flags.append(1.0 if item.get("success", False) else 0.0)
                progress_path.write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "task": args.task,
                            "split": args.split,
                            "method": args.method,
                            "hold_steps": int(args.hold_steps),
                            "control_loop": bool(args.control_loop),
                            "completed_packs": len(results),
                            "total_packs": len(pack_paths),
                            "partial_sr": float(np.mean(success_flags)) if success_flags else None,
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
        "method": args.method,
        "hold_steps": int(args.hold_steps),
        "control_loop": bool(args.control_loop),
        "pack_dir": str(pack_dir),
        "num_packs": len(pack_paths),
        "num_success": int(sum(1 for x in success_flags if x > 0.5)),
        "sr": float(np.mean(success_flags)) if success_flags else 0.0,
        "elapsed_sec": float(time.perf_counter() - start),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
