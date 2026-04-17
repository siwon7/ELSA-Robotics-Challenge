import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rlbench.backend.exceptions import InvalidActionError

from elsa_learning_agent.utils import (
    denormalize_action,
    get_image_transform,
    prepare_action_for_env,
    process_obs,
)
from federated_elsa_robotics.eval_model import (
    build_policy_input_adapter,
    build_split_config,
    build_live_env_ids,
    is_legacy_bc_agent,
    legacy_bc_load_environment,
    legacy_bc_process_obs,
    load_agent,
    load_environment,
    write_episode_videos,
)


def round_list(values, digits=6):
    return [round(float(v), digits) for v in values]


def main():
    parser = argparse.ArgumentParser(description="Trace a live rollout and save pose/action data.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-config-path", default="dataset_config.yaml")
    parser.add_argument("--policy-name", default=None)
    parser.add_argument("--split", default="eval", choices=["eval", "test"])
    parser.add_argument("--env-id", type=int, default=400)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-video-dir", default=None)
    parser.add_argument("--ik-solver", default=None)
    args = parser.parse_args()

    if args.ik_solver:
        os.environ["ELSA_IK_SOLVER"] = args.ik_solver

    cfg = OmegaConf.load(args.dataset_config_path)
    split_cfg = build_split_config(cfg, args.task, args.split, args.env_id)
    fed_cfg = OmegaConf.load(
        os.path.join(split_cfg.dataset.root_dir, args.task, f"{args.task}_fed.yaml")
    )
    split_cfg.env = fed_cfg.env
    split_cfg.data = fed_cfg.data
    split_cfg.data.renderer = os.environ.get("ELSA_SIM_RENDERER", "opengl")

    transform = get_image_transform(split_cfg)
    agent = load_agent(
        args.model_path,
        args.device,
        config=split_cfg,
        policy_name=args.policy_name,
    )
    input_adapter = build_policy_input_adapter(agent, split_cfg, args.device)

    collection_cfg_path = os.path.join(
        split_cfg.dataset.root_dir,
        args.task,
        f"{args.task}_fed.json",
    )
    with open(collection_cfg_path, "r", encoding="utf-8") as fh:
        collection_cfg = json.load(fh)

    if is_legacy_bc_agent(agent):
        task_env, rlbench_env = legacy_bc_load_environment(
            split_cfg,
            collection_cfg,
            args.env_id,
            headless=args.headless,
        )
    else:
        task_env, rlbench_env = load_environment(
            split_cfg,
            collection_cfg,
            args.env_id,
            headless=args.headless,
        )

    frames = []
    trace = []
    try:
        for _ in range(args.episode_index + 1):
            _, obs = task_env.reset()
        frames.append(obs.front_rgb.copy())
        terminate = False
        reward = 0.0
        step = 0
        episode_error = None
        while not terminate and step < args.max_steps:
            with torch.no_grad():
                if is_legacy_bc_agent(agent):
                    front_rgb, low_dim_state = legacy_bc_process_obs(obs, transform)
                else:
                    front_rgb, low_dim_state = process_obs(obs, transform)
                front_rgb = front_rgb.unsqueeze(0).to(args.device)
                if input_adapter is not None:
                    front_rgb = input_adapter(front_rgb).float().clone()
                low_dim_state = low_dim_state.unsqueeze(0).to(args.device)
                predicted_action = agent.get_action(front_rgb, low_dim_state)

            normalized_action = predicted_action.detach().cpu().numpy()[0]
            denormalized_action = denormalize_action(
                predicted_action.detach().cpu(),
                torch.tensor(split_cfg.transform.action_min),
                torch.tensor(split_cfg.transform.action_max),
            ).numpy()[0]
            env_action = prepare_action_for_env(denormalized_action, split_cfg)

            current_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
            target_pose = np.asarray(env_action[:8], dtype=np.float32)
            pos_delta = target_pose[:3] - current_pose[:3]

            try:
                obs, reward, terminate = task_env.step(env_action)
                frames.append(obs.front_rgb.copy())
                next_pose = round_list(np.asarray(obs.gripper_pose, dtype=np.float32))
            except InvalidActionError as exc:
                episode_error = str(exc)
                reward = 0.0
                terminate = True
                next_pose = None

            trace.append(
                {
                    "step": step,
                    "reward": float(reward),
                    "terminate": bool(terminate),
                    "normalized_action": round_list(normalized_action),
                    "denormalized_action": round_list(denormalized_action),
                    "current_pose": round_list(current_pose),
                    "target_pose": round_list(target_pose),
                    "next_pose": next_pose,
                    "pos_delta_xyz": round_list(pos_delta),
                    "pos_delta_norm": round(float(np.linalg.norm(pos_delta)), 6),
                    "error": episode_error,
                }
            )
            step += 1
    finally:
        rlbench_env.shutdown()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": args.task,
        "split": args.split,
        "env_id": args.env_id,
        "episode_index": args.episode_index,
        "device": args.device,
        "ik_solver": os.environ.get("ELSA_IK_SOLVER", "jacobian"),
        "model_path": args.model_path,
        "policy_name": args.policy_name,
        "num_steps": len(trace),
        "final_reward": float(reward),
        "terminated": bool(terminate),
        "error": episode_error,
        "live_env_ids": build_live_env_ids(cfg, args.split),
        "trace": trace,
    }

    if args.save_video_dir:
        video_base = Path(args.save_video_dir) / (
            f"{args.task}_{args.split}_env_{args.env_id}_episode_{args.episode_index}"
        )
        video_base.parent.mkdir(parents=True, exist_ok=True)
        result["video_paths"] = write_episode_videos(frames, video_base, fps=20)

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
