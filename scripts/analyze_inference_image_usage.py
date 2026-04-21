import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.utils import (
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
)


def tensor_stats(x: torch.Tensor) -> dict:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-config-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.dataset_config_path)
    cfg.dataset.task = args.task
    cfg.dataset.env_id = int(args.env_id)
    cfg.dataset.batch_size = int(args.batch_size)
    cfg.dataset.num_workers = 0
    if "train_split" not in cfg.dataset:
        cfg.dataset.train_split = 0.9
    if "test_split" not in cfg.dataset:
        cfg.dataset.test_split = float(cfg.dataset.train_split)

    dataset = ImitationDataset(
        cfg,
        train=args.split == "train",
        test=args.split == "test",
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    sample = next(iter(loader))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = Agent(
        image_channels=3,
        low_dim_state_dim=sample["low_dim_state"].shape[1],
        action_dim=sample["action"].shape[1],
        image_size=(sample["image"].shape[2], sample["image"].shape[3]),
        **get_agent_model_kwargs(cfg),
    )
    state_dict = torch.load(args.model_path, map_location=device)
    agent.policy.load_state_dict(state_dict)
    agent.policy.to(device)
    agent.eval()

    zero_image_deltas = []
    shuffle_image_deltas = []
    zero_state_deltas = []
    shuffle_state_deltas = []
    base_pred_std = []
    saturation_fractions = []
    image_embed_norms = []
    state_embed_norms = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            image = batch["image"].to(device)
            low_dim = batch["low_dim_state"].to(device)

            pred = agent.policy(image, low_dim)
            pred_zero_image = agent.policy(torch.zeros_like(image), low_dim)
            pred_zero_state = agent.policy(image, torch.zeros_like(low_dim))

            if image.shape[0] > 1:
                image_perm = torch.randperm(image.shape[0], device=device)
                low_dim_perm = torch.randperm(low_dim.shape[0], device=device)
                pred_shuffle_image = agent.policy(image[image_perm], low_dim)
                pred_shuffle_state = agent.policy(image, low_dim[low_dim_perm])
            else:
                pred_shuffle_image = pred.clone()
                pred_shuffle_state = pred.clone()

            zero_image_deltas.append(torch.norm(pred - pred_zero_image, dim=-1).mean().item())
            shuffle_image_deltas.append(torch.norm(pred - pred_shuffle_image, dim=-1).mean().item())
            zero_state_deltas.append(torch.norm(pred - pred_zero_state, dim=-1).mean().item())
            shuffle_state_deltas.append(torch.norm(pred - pred_shuffle_state, dim=-1).mean().item())
            base_pred_std.append(pred.std(dim=0, unbiased=False).mean().item())
            saturation_fractions.append((pred.abs() > 0.99).float().mean().item())

            image_embed = agent.policy.cnn_encoder(image)
            state_embed = agent.policy.mlp_encoder(low_dim)
            image_embed_norms.append(image_embed.norm(dim=-1).mean().item())
            state_embed_norms.append(state_embed.norm(dim=-1).mean().item())

    payload = {
        "model_path": args.model_path,
        "dataset_config_path": args.dataset_config_path,
        "task": args.task,
        "env_id": int(args.env_id),
        "split": args.split,
        "num_samples": int(min(len(dataset), args.batch_size * args.num_batches)),
        "vision_backbone": str(getattr(cfg.model, "vision_backbone", "cnn")),
        "action_pipeline_preset": str(get_action_pipeline_preset(cfg)),
        "action_representation": str(get_action_representation(cfg)),
        "execution_action_interface": str(get_execution_action_interface(cfg)),
        "execution_action_adapter": str(get_execution_action_adapter(cfg)),
        "action_output_activation": get_agent_model_kwargs(cfg)["action_output_activation"],
        "metrics": {
            "base_prediction": tensor_stats(torch.tensor(base_pred_std)),
            "zero_image_l2_delta": tensor_stats(torch.tensor(zero_image_deltas)),
            "shuffle_image_l2_delta": tensor_stats(torch.tensor(shuffle_image_deltas)),
            "zero_state_l2_delta": tensor_stats(torch.tensor(zero_state_deltas)),
            "shuffle_state_l2_delta": tensor_stats(torch.tensor(shuffle_state_deltas)),
            "saturation_fraction": tensor_stats(torch.tensor(saturation_fractions)),
            "image_embedding_norm": tensor_stats(torch.tensor(image_embed_norms)),
            "state_embedding_norm": tensor_stats(torch.tensor(state_embed_norms)),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
