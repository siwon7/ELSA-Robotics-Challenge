#!/usr/bin/env python3
"""Offline validation for Flower checkpoints with client-local parameter state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.config_validation import validate_runtime_config
from elsa_learning_agent.dataset.path_utils import resolve_dataset_root
from federated_elsa_robotics.fl_method_registry import resolve_prox_mu
from federated_elsa_robotics.parameter_surfaces import (
    load_local_only_state,
    uses_local_parameter_state,
)
from federated_elsa_robotics.task import (
    infer_action_dim,
    load_data_colosseum,
    validate_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a server checkpoint plus optional FedPer/FedRep-style local "
            "parameter state and run offline validation for one partition/env."
        )
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-root", default="model_checkpoints")
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=400)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--split", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prox-mu", default="")
    parser.add_argument("--local-state-path", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def normalize_run_tag(raw_tag) -> str:
    tag = str(raw_tag or "").strip()
    if not tag:
        return "default"
    return tag.replace(" ", "-")


def load_torch_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def configure_dataset(config, args: argparse.Namespace):
    config.dataset.dataset_task = args.task
    config.dataset.task = args.task
    config.dataset.env_id = int(args.env_id)
    if args.split == "train":
        root = config.dataset.root_dir
    elif args.split == "eval":
        root = config.dataset.root_eval_dir
    else:
        root = config.dataset.root_test_dir
    config.dataset.root_dir = resolve_dataset_root(str(root), args.task)
    config.dataset.test_split = 0.0
    config.dataset.action_dim = infer_action_dim(config)
    return config


def default_local_state_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.checkpoint_root)
        / str(args.task)
        / "client_local_state"
        / normalize_run_tag(args.run_tag)
        / f"partition_{int(args.partition_id)}_env_{int(args.env_id)}.pt"
    )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    config = OmegaConf.load(args.dataset_config_path)
    config = configure_dataset(config, args)
    config.model.prox_mu = resolve_prox_mu(config, explicit_override=args.prox_mu)
    validation_summary = validate_runtime_config(config)

    trainloader, valloader = load_data_colosseum(
        args.partition_id,
        args.num_partitions,
        config=config,
    )
    sample = next(iter(trainloader))
    sample_action_dim = int(sample["action"].shape[1])
    action_dim = int(config.dataset.action_dim)
    if action_dim != sample_action_dim:
        action_dim = sample_action_dim

    agent = Agent(
        image_channels=int(sample["image"].shape[1]),
        low_dim_state_dim=sample["low_dim_state"].shape[1],
        action_dim=action_dim,
        image_size=(sample["image"].shape[2], sample["image"].shape[3]),
        **get_agent_model_kwargs(config),
    )
    agent.policy.to(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_state = load_torch_state(checkpoint_path, device)
    missing, unexpected = agent.policy.load_state_dict(checkpoint_state, strict=False)

    loaded_local_tensors = 0
    local_state_path = Path(args.local_state_path) if args.local_state_path else None
    if uses_local_parameter_state(config):
        local_state_path = local_state_path or default_local_state_path(args)
        loaded_local_tensors = load_local_only_state(
            agent,
            config,
            local_state_path,
            device,
        )

    val_loss = validate_one_epoch(agent, valloader, device)
    payload = {
        "task": args.task,
        "split": args.split,
        "partition_id": int(args.partition_id),
        "env_id": int(args.env_id),
        "checkpoint": str(checkpoint_path),
        "local_state_path": str(local_state_path) if local_state_path else "",
        "loaded_local_tensors": int(loaded_local_tensors),
        "val_loss": float(val_loss),
        "missing_checkpoint_keys": list(missing),
        "unexpected_checkpoint_keys": list(unexpected),
        "validation_summary": validation_summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
