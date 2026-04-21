#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.config_validation import validate_runtime_config
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.dataset.path_utils import resolve_dataset_root
from federated_elsa_robotics.task import infer_action_dim


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate config/model/dataloader/action compatibility on CPU."
    )
    parser.add_argument("--config", required=True, help="Path to experiment or dataset yaml.")
    parser.add_argument("--task", default=None, help="Override dataset.task")
    parser.add_argument("--env-id", type=int, default=None, help="Override dataset.env_id")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split to probe",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Load dataset with action normalization enabled.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config = OmegaConf.load(repo_root / "dataset_config.yaml")
    override_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_config, override_config)
    if args.task is not None:
        config.dataset.task = args.task
    if args.env_id is not None:
        config.dataset.env_id = int(args.env_id)
    config.dataset.root_dir = resolve_dataset_root(
        str(config.dataset.root_dir), str(config.dataset.task)
    )
    config.dataset.root_eval_dir = resolve_dataset_root(
        str(config.dataset.root_eval_dir), str(config.dataset.task)
    )
    config.dataset.root_test_dir = resolve_dataset_root(
        str(config.dataset.root_test_dir), str(config.dataset.task)
    )

    summary = validate_runtime_config(config)
    config.dataset.action_dim = infer_action_dim(config)

    dataset = ImitationDataset(
        config=config,
        train=args.split == "train",
        test=args.split == "test",
        normalize=args.normalize,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty for the selected split.")

    sample = dataset[0]
    image = sample["image"].unsqueeze(0)
    low_dim_state = sample["low_dim_state"].unsqueeze(0)
    action = sample["action"].unsqueeze(0)

    agent = Agent(
        image_channels=int(image.shape[1]),
        low_dim_state_dim=int(low_dim_state.shape[1]),
        action_dim=int(action.shape[1]),
        image_size=(int(image.shape[2]), int(image.shape[3])),
        **get_agent_model_kwargs(config),
    )
    agent.eval()
    with torch.no_grad():
        predicted = agent.get_action(image, low_dim_state)

    report = {
        "config": args.config,
        "task": str(config.dataset.task),
        "env_id": int(config.dataset.env_id),
        "split": args.split,
        "normalize": bool(args.normalize),
        "dataset_size": len(dataset),
        "image_shape": list(image.shape),
        "low_dim_shape": list(low_dim_state.shape),
        "action_shape": list(action.shape),
        "predicted_shape": list(predicted.shape),
        "validation_summary": summary,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
