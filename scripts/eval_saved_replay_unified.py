"""Unified replay-ceiling probe across all RLBench-supported action modes.

Supports arm modes:
  jv                       - JointVelocity (uses replay_actions['stored_joint_vel'])
  jv_finite_diff           - JointVelocity (uses replay_actions['finite_diff'])
  jp_abs                   - JointPosition(absolute_mode=True), action = next_q
  jp_rel                   - JointPosition(absolute_mode=False), action = next_q - current_q
  ee_ik_abs_world[_coll]   - EndEffectorPoseViaIK, abs, world frame
  ee_ik_rel_world[_coll]   - EndEffectorPoseViaIK, rel, world frame
  ee_ik_abs_ee[_coll]      - EndEffectorPoseViaIK, abs, EE frame
  ee_ik_rel_ee[_coll]      - EndEffectorPoseViaIK, rel, EE frame
  ee_plan_abs_world[_coll] - EndEffectorPoseViaPlanning, abs, world frame
  ee_plan_rel_world[_coll] - EndEffectorPoseViaPlanning, rel, world frame

For EE modes, end-effector pose at each saved joint configuration is recovered via
forward kinematics: the arm is briefly moved to q_t kinematically (disable_dynamics),
the tip pose is queried, then the arm is restored to its physics state.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from elsa_learning_agent.utils import load_environment
from scripts.live_eval_common import deserialize_random_state, load_main_split_cfg

ARM_MODE_TO_INTERFACE = {
    "jv": "joint_velocity",
    "jv_finite_diff": "joint_velocity",
    "jp_abs": "joint_position",
    "jp_rel": "joint_position_relative",
    "ee_ik_abs_world": "ee_pose_ik_abs_world",
    "ee_ik_rel_world": "ee_pose_ik_rel_world",
    "ee_ik_abs_ee": "ee_pose_ik_abs_ee",
    "ee_ik_rel_ee": "ee_pose_ik_rel_ee",
    "ee_ik_abs_world_coll": "ee_pose_ik_abs_world_collision",
    "ee_ik_rel_world_coll": "ee_pose_ik_rel_world_collision",
    "ee_plan_abs_world": "ee_pose_plan_abs_world",
    "ee_plan_rel_world": "ee_pose_plan_rel_world",
    "ee_plan_abs_world_coll": "ee_pose_plan_abs_world_collision",
}


def is_ee_mode(arm_mode: str) -> bool:
    return arm_mode.startswith("ee_")


def is_relative(arm_mode: str) -> bool:
    return "_rel_" in arm_mode or arm_mode == "jp_rel"


def fk_pose_at_joints(arm, q):
    """Forward-kinematics: temporarily set arm to q, read tip pose (xyz + quat), restore."""
    saved_q = np.asarray(arm.get_joint_positions(), dtype=np.float32)
    arm.set_joint_positions(np.asarray(q, dtype=np.float32), disable_dynamics=True)
    pose = np.asarray(arm.get_tip().get_pose(), dtype=np.float32)  # 7: x y z qx qy qz qw
    arm.set_joint_positions(saved_q, disable_dynamics=True)
    return pose


def relative_pose_world(p_curr, p_next):
    """Delta in world frame: just position delta + quat delta (next * curr.inv())."""
    from pyrep.objects.object import Object  # ensure pyrep loaded
    # For EE mode with absolute_mode=False and frame='world', RLBench expects the
    # action to be (delta_xyz, delta_quat) where delta_quat is the quaternion that
    # rotates from current to target in world frame: q_delta = q_next * q_curr^{-1}.
    pos_delta = p_next[:3] - p_curr[:3]
    qx, qy, qz, qw = p_curr[3], p_curr[4], p_curr[5], p_curr[6]
    # quaternion conjugate (inverse for unit quat) of curr
    q_curr_inv = np.array([-qx, -qy, -qz, qw], dtype=np.float32)
    q_delta = quat_mul(p_next[3:7], q_curr_inv)
    return np.concatenate([pos_delta, q_delta]).astype(np.float32)


def relative_pose_ee_frame(p_curr, p_next):
    """Delta expressed in current EE frame: T_delta = T_curr.inv() @ T_next."""
    qx, qy, qz, qw = p_curr[3], p_curr[4], p_curr[5], p_curr[6]
    q_curr_inv = np.array([-qx, -qy, -qz, qw], dtype=np.float32)
    pos_delta_world = p_next[:3] - p_curr[:3]
    pos_delta_ee = quat_rotate(q_curr_inv, pos_delta_world)
    q_delta_ee = quat_mul(q_curr_inv, p_next[3:7])
    return np.concatenate([pos_delta_ee, q_delta_ee]).astype(np.float32)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (xyzw)."""
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    qx, qy, qz, qw = q
    q_inv = np.array([-qx, -qy, -qz, qw], dtype=np.float32)
    rot = quat_mul(quat_mul(q, qv), q_inv)
    return rot[:3]


def build_action_for_step(arm_mode, arm, jp_curr, jp_next, gripper_next, replay_actions, t):
    """Build an arm-mode-specific action of shape (arm_dim + 1,) including gripper."""
    if arm_mode == "jv":
        arm_act = np.asarray(replay_actions["stored_joint_vel"][t], dtype=np.float32)
    elif arm_mode == "jv_finite_diff":
        arm_act = np.asarray(replay_actions["finite_diff"][t], dtype=np.float32)
    elif arm_mode == "jp_abs":
        arm_act = np.asarray(jp_next, dtype=np.float32)
    elif arm_mode == "jp_rel":
        arm_act = np.asarray(jp_next - jp_curr, dtype=np.float32)
    elif is_ee_mode(arm_mode):
        p_next = fk_pose_at_joints(arm, jp_next)
        if "_abs_" in arm_mode:
            arm_act = p_next
        else:
            p_curr = fk_pose_at_joints(arm, jp_curr)
            if "_world" in arm_mode:
                arm_act = relative_pose_world(p_curr, p_next)
            else:  # _ee
                arm_act = relative_pose_ee_frame(p_curr, p_next)
    else:
        raise ValueError(f"Unknown arm_mode: {arm_mode}")
    return np.concatenate([arm_act, [float(gripper_next)]]).astype(np.float32)


def build_replay_env(task_name, split, env_id, arm_mode):
    cfg, collection_cfg = load_main_split_cfg(task_name, split)
    cfg.dataset.execution_action_interface = ARM_MODE_TO_INTERFACE[arm_mode]
    cfg.dataset.execution_action_adapter = "none"
    if is_ee_mode(arm_mode):
        cfg.dataset.action_representation = "end_effector_pose"
    elif arm_mode in ("jp_abs",):
        cfg.dataset.action_representation = "joint_position_absolute"
    elif arm_mode == "jp_rel":
        cfg.dataset.action_representation = "joint_position_relative"
    else:
        cfg.dataset.action_representation = "joint_velocity"
    task_env, rlbench_env = load_environment(cfg, collection_cfg, env_id, headless=True)
    return cfg, rlbench_env, task_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="training", choices=["training", "eval", "test"])
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--arm-mode", required=True, choices=list(ARM_MODE_TO_INTERFACE.keys()))
    parser.add_argument("--hold-steps", type=int, default=1)
    parser.add_argument("--max-pack-time-sec", type=float, default=120.0,
                        help="Per-pack timeout to bail on slow modes (e.g. EE Planning).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pack_paths = sorted(Path(args.pack_dir).glob("*.replay.pkl"))
    if not pack_paths:
        raise FileNotFoundError(f"no replay packs found in {args.pack_dir}")

    by_env = {}
    for pack_path in pack_paths:
        with open(pack_path, "rb") as fh:
            pack = pickle.load(fh)
        by_env.setdefault(int(pack["env_id"]), []).append((pack_path, pack))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")

    success_flags = []
    results = []
    start = time.perf_counter()

    infeasible_init = 0
    for env_id in sorted(by_env):
        try:
            cfg, rlbench_env, task_env = build_replay_env(args.task, args.split, env_id, args.arm_mode)
        except Exception as e:
            print(f"  env {env_id}: build_replay_env failed: {type(e).__name__}: {e}")
            for pack_path, pack in by_env[env_id]:
                success_flags.append(0)
                results.append({
                    "env_id": int(pack["env_id"]),
                    "demo_idx": int(pack["demo_idx"]),
                    "reward": 0.0,
                    "terminated": False,
                    "success": False,
                    "infeasible_init": True,
                    "init_error": f"{type(e).__name__}: {e}",
                    "pack_path": str(pack_path),
                })
                infeasible_init += 1
            continue
        try:
            arm = task_env._robot.arm
            for pack_path, pack in by_env[env_id]:
                random_state = deserialize_random_state(pack.get("random_seed"))
                if random_state is None:
                    raise ValueError("pack does not include random_seed")
                np.random.set_state(random_state)
                try:
                    _descriptions, _obs = task_env.reset()
                except Exception as e:
                    print(f"  pack {pack_path.name}: reset failed: {type(e).__name__}: {e}")
                    success_flags.append(0)
                    results.append({
                        "env_id": int(pack["env_id"]),
                        "demo_idx": int(pack["demo_idx"]),
                        "reward": 0.0,
                        "terminated": False,
                        "success": False,
                        "infeasible_init": True,
                        "init_error": f"{type(e).__name__}: {e}",
                        "pack_path": str(pack_path),
                    })
                    infeasible_init += 1
                    continue
                seq_q = pack["joint_positions"]
                seq_grip = pack["gripper_open"]
                replay_actions = pack.get("replay_actions", {})
                reward = 0.0
                terminated = False
                pack_start = time.perf_counter()
                steps = 0
                ik_fail = 0
                try:
                    for t in range(len(seq_q) - 1):
                        if time.perf_counter() - pack_start > args.max_pack_time_sec:
                            print(f"  TIMEOUT at step {t}/{len(seq_q)-1} on {pack_path.name}")
                            break
                        jp_curr = np.asarray(seq_q[t], dtype=np.float32)
                        jp_next = np.asarray(seq_q[t + 1], dtype=np.float32)
                        try:
                            action = build_action_for_step(
                                args.arm_mode, arm, jp_curr, jp_next, seq_grip[t + 1], replay_actions, t
                            )
                        except KeyError as e:
                            raise RuntimeError(
                                f"arm_mode {args.arm_mode} requires replay_actions key {e}"
                            )
                        for _ in range(max(1, int(args.hold_steps))):
                            try:
                                _obs, step_reward, terminate = task_env.step(action)
                                reward = float(step_reward)
                                steps += 1
                                if terminate:
                                    terminated = True
                                    break
                            except Exception as e:
                                ik_fail += 1
                                if ik_fail > 50:
                                    print(f"  too many IK/exec failures on {pack_path.name}")
                                    raise
                                break
                        if terminated:
                            break
                except Exception as e:
                    print(f"  pack {pack_path.name} failed: {type(e).__name__}: {e}")
                success = terminated or (reward > 0.0)
                success_flags.append(int(success))
                rec = {
                    "env_id": int(pack["env_id"]),
                    "demo_idx": int(pack["demo_idx"]),
                    "reward": reward,
                    "terminated": bool(terminated),
                    "success": bool(success),
                    "num_actions": int(len(seq_q) - 1),
                    "steps": int(steps),
                    "ik_fail": int(ik_fail),
                    "pack_path": str(pack_path),
                }
                results.append(rec)
                with open(progress_path, "w") as f:
                    json.dump(
                        {
                            "task": args.task,
                            "arm_mode": args.arm_mode,
                            "completed": len(results),
                            "total": len(pack_paths),
                            "running_sr": float(np.mean(success_flags)) if success_flags else 0.0,
                            "elapsed_sec": time.perf_counter() - start,
                        },
                        f,
                        indent=2,
                    )
        finally:
            try:
                rlbench_env.shutdown()
            except Exception:
                pass

    elapsed = time.perf_counter() - start
    sr = float(np.mean(success_flags)) if success_flags else 0.0
    out = {
        "task": args.task,
        "split": args.split,
        "arm_mode": args.arm_mode,
        "execution_action_interface": ARM_MODE_TO_INTERFACE[args.arm_mode],
        "hold_steps": args.hold_steps,
        "num_packs": len(pack_paths),
        "num_success": int(sum(success_flags)),
        "num_infeasible_init": int(infeasible_init),
        "sr": sr,
        "elapsed_sec": elapsed,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.task}][{args.arm_mode}] sr={sr:.3f} ({sum(success_flags)}/{len(success_flags)}) in {elapsed:.1f}s")
    if progress_path.exists():
        try:
            progress_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
