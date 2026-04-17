import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from colosseum import TASKS_PY_FOLDER, TASKS_TTM_FOLDER
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

from elsa_learning_agent.dataset.keypoint_discovery import discover_heuristic_keypoints
from scripts.live_eval_common import deserialize_random_state, load_main_split_cfg


class MoveArmThenGripperJointPosition(ActionMode):
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


class _PackObs:
    def __init__(self, joint_velocities, gripper_open):
        self.joint_velocities = joint_velocities
        self.gripper_open = gripper_open


def load_joint_position_environment(task_name: str, split: str, env_id: int):
    cfg, collection_cfg = load_main_split_cfg(task_name, split)
    task = name_to_class(cfg.env.task_name, TASKS_PY_FOLDER)
    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    env_entry = next(
        entry for entry in collection_cfg["env_config"] if entry["env_idx"] == env_id
    )
    for variation_cfg in env_entry.get("variations_parameters", []):
        var_type = variation_cfg["type"]
        var_name = variation_cfg.get("name")
        for factor_cfg in config.env.scene.factors:
            if factor_cfg.variation != var_type:
                continue
            if var_name is not None and "name" in factor_cfg and factor_cfg.name != var_name:
                continue
            for key, value in variation_cfg.items():
                if key == "type":
                    continue
                factor_cfg[key] = value
            break

    obs_config = ObservationConfigExt(config.data)
    obs_config.record_gripper_closing = True
    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripperJointPosition(
            arm_action_mode=JointPosition(absolute_mode=True),
            gripper_action_mode=Discrete(),
        ),
        obs_config=obs_config,
        headless=True,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=config.env,
    )
    rlbench_env.launch()
    return rlbench_env, rlbench_env.get_task(task)


def select_keyframe_indices(pack, method: str, fixed_horizon: int, stopping_delta: float, stopped_buffer_steps: int):
    seq_len = len(pack["joint_positions"])
    if seq_len < 2:
        return [0]
    if method == "fixed_horizon":
        indices = list(range(0, seq_len, max(1, fixed_horizon)))
        if indices[-1] != seq_len - 1:
            indices.append(seq_len - 1)
        return indices
    if method == "peract_heuristic":
        pseudo_traj = [
            _PackObs(joint_velocities=pack["joint_velocities"][i], gripper_open=pack["gripper_open"][i])
            for i in range(seq_len)
        ]
        indices = [0] + discover_heuristic_keypoints(
            pseudo_traj,
            stopping_delta=stopping_delta,
            stopped_buffer_steps=stopped_buffer_steps,
        )
        deduped = []
        for idx in indices:
            idx = int(idx)
            if not deduped or deduped[-1] != idx:
                deduped.append(idx)
        return deduped
    raise ValueError(f"Unsupported method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--method", required=True, choices=["fixed_horizon", "peract_heuristic"])
    parser.add_argument("--fixed-horizon", type=int, default=4)
    parser.add_argument("--hold-steps", type=int, default=1)
    parser.add_argument("--stopping-delta", type=float, default=0.1)
    parser.add_argument("--stopped-buffer-steps", type=int, default=4)
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
        rlbench_env, task_env = load_joint_position_environment(args.task, args.split, env_id)
        try:
            for pack_path, pack in by_env[env_id]:
                random_state = deserialize_random_state(pack.get("random_seed"))
                if random_state is None:
                    raise ValueError("pack does not include random_seed")
                key_indices = select_keyframe_indices(
                    pack=pack,
                    method=args.method,
                    fixed_horizon=args.fixed_horizon,
                    stopping_delta=args.stopping_delta,
                    stopped_buffer_steps=args.stopped_buffer_steps,
                )
                np.random.set_state(random_state)
                task_env.reset()
                reward = 0.0
                terminated = False
                steps = 0
                for idx in key_indices[1:]:
                    action = np.concatenate(
                        (
                            np.asarray(pack["joint_positions"][idx], dtype=np.float32),
                            np.asarray([pack["gripper_open"][idx]], dtype=np.float32),
                        ),
                        axis=0,
                    )
                    for _ in range(max(1, int(args.hold_steps))):
                        _obs, step_reward, terminate = task_env.step(action)
                        reward = float(step_reward)
                        steps += 1
                        if terminate:
                            terminated = True
                            break
                    if terminated:
                        break
                success = terminated or (reward > 0.0)
                success_flags.append(1.0 if success else 0.0)
                results.append(
                    {
                        "env_id": env_id,
                        "demo_idx": int(pack["demo_idx"]),
                        "success": success,
                        "reward": reward,
                        "terminated": terminated,
                        "num_keyframes": len(key_indices),
                        "keyframe_indices": key_indices,
                        "steps": steps,
                        "pack_path": str(pack_path),
                    }
                )
        finally:
            rlbench_env.shutdown()

    payload = {
        "task": args.task,
        "split": args.split,
        "method": args.method,
        "fixed_horizon": int(args.fixed_horizon),
        "hold_steps": int(args.hold_steps),
        "stopping_delta": float(args.stopping_delta),
        "stopped_buffer_steps": int(args.stopped_buffer_steps),
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
