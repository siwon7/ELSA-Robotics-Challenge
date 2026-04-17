import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from elsa_learning_agent.agent_forward_kinematics import infer_policy_name_from_model_path
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from federated_elsa_robotics.eval_model import load_agent
from federated_elsa_robotics.policy_runtime import trim_low_dim_state


def main():
    parser = argparse.ArgumentParser(description="Summarize action distribution for a checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset-config-path", default="dataset_config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-name", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.dataset_config_path)
    cfg.dataset.root_dir = cfg.dataset.root_eval_dir
    cfg.dataset.task = args.task
    cfg.dataset.env_id = args.env_id
    cfg.dataset.train_split = 0.9

    model_path = Path(args.model_path)
    policy_name = args.policy_name or infer_policy_name_from_model_path(
        str(model_path), default=cfg.model.get("policy_name")
    )
    cfg.model.policy_name = policy_name

    agent = load_agent(str(model_path), args.device, config=cfg, policy_name=policy_name)
    dataset = ImitationDataset(config=cfg, train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    preds = []
    actions = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(args.device)
            low_dim_state = trim_low_dim_state(agent, batch["low_dim_state"].to(args.device))
            pred = agent.get_action(image, low_dim_state).detach().cpu().numpy()
            preds.append(pred)
            actions.append(batch["action"].numpy())

    preds = np.concatenate(preds, axis=0)
    actions = np.concatenate(actions, axis=0)

    result = {
        "task": args.task,
        "env_id": args.env_id,
        "model_path": str(model_path),
        "policy_name": policy_name,
        "num_samples": int(len(preds)),
        "pred_std": np.round(preds.std(axis=0), 6).tolist(),
        "pred_mean": np.round(preds.mean(axis=0), 6).tolist(),
        "target_std": np.round(actions.std(axis=0), 6).tolist(),
        "target_mean": np.round(actions.mean(axis=0), 6).tolist(),
        "gripper_acc_thresh_0_5": float(
            ((preds[:, -1] > 0.5) == (actions[:, -1] > 0.5)).mean()
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
