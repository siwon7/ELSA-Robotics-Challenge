import argparse
import json
import math
import os
from pathlib import Path

from omegaconf import OmegaConf

from elsa_learning_agent.agent_forward_kinematics import (
    infer_policy_name_from_model_path,
    policy_uses_cached_visual_features,
)
from federated_elsa_robotics.eval_model import evaluate_offline, evaluate_online, load_agent


def infer_metadata(model_path: Path):
    name = model_path.name
    task = model_path.parent.name
    local_epochs = None
    round_num = None
    for part in name.split("_"):
        if part == "l-ep":
            continue
    marker = "l-ep_"
    if marker in name:
        local_epochs = int(name.split(marker, 1)[1].split("_", 1)[0])
    round_marker = "_round_"
    if round_marker in name:
        round_num = int(name.split(round_marker, 1)[1].split(".pth", 1)[0])
    return task, local_epochs, round_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--split", default="eval", choices=["eval", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--policy-name", default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    inferred_task, local_epochs, round_num = infer_metadata(model_path)
    task = args.task or inferred_task

    cfg = OmegaConf.load("dataset_config.yaml")
    policy_name = args.policy_name or infer_policy_name_from_model_path(
        str(model_path),
        default=cfg.model.get("policy_name"),
    )
    cfg.model.policy_name = policy_name
    if policy_uses_cached_visual_features(policy_name):
        cfg.model.cache_features = True
    agent = load_agent(str(model_path), args.device, config=cfg, policy_name=policy_name)
    offline = evaluate_offline(
        agent=agent,
        base_config=cfg,
        task=task,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    result = {
        "task": task,
        "model_path": str(model_path),
        "policy_name": policy_name,
        "local_epochs": local_epochs,
        "round": round_num,
        "mse": offline["mean_loss"],
        "rmse": math.sqrt(offline["mean_loss"]),
        "std_mse": offline["std_loss"],
        "offline": offline,
    }

    if args.simulator:
        online = evaluate_online(
            agent=agent,
            base_config=cfg,
            task=task,
            split=args.split,
            device=args.device,
        )
        result["sr"] = online["mean_reward"]
        result["std_sr"] = online["std_reward"]
        result["online"] = online

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
