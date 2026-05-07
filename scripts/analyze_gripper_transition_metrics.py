#!/usr/bin/env python3
from __future__ import annotations

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
from elsa_learning_agent.dataset.path_utils import resolve_dataset_root
from elsa_learning_agent.utils import move_nested_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure gripper open/close prediction quality near expert transition frames."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML to evaluate.")
    parser.add_argument("--checkpoint", default=None, help="Optional policy checkpoint path.")
    parser.add_argument("--task", default=None, help="Override dataset.task.")
    parser.add_argument("--env-id", type=int, default=None, help="Override dataset.env_id.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transition-window", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    return parser.parse_args()


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.bce_sum = 0.0
        self.correct = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, prob_open: torch.Tensor, target_open: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        if not bool(mask.any()):
            return
        prob = prob_open[mask].detach().float().clamp(1e-6, 1.0 - 1e-6)
        target = target_open[mask].detach().float()
        pred = prob >= 0.5
        target_bool = target >= 0.5
        bce = -target * torch.log(prob) - (1.0 - target) * torch.log(1.0 - prob)

        self.count += int(target.numel())
        self.bce_sum += float(bce.sum().item())
        self.correct += int((pred == target_bool).sum().item())
        self.tp += int((pred & target_bool).sum().item())
        self.fp += int((pred & ~target_bool).sum().item())
        self.tn += int((~pred & ~target_bool).sum().item())
        self.fn += int((~pred & target_bool).sum().item())

    def summary(self) -> dict:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        return {
            "count": int(self.count),
            "bce": self.bce_sum / max(1, self.count),
            "accuracy": self.correct / max(1, self.count),
            "precision_open": precision,
            "recall_open": recall,
            "f1_open": f1,
            "tp": int(self.tp),
            "fp": int(self.fp),
            "tn": int(self.tn),
            "fn": int(self.fn),
        }


def load_config(config_path: str, task: str | None, env_id: int | None, transition_window: int):
    repo_root = Path(__file__).resolve().parents[1]
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    base = OmegaConf.load(repo_root / "dataset_config.yaml")
    override = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base, override)
    if task is not None:
        cfg.dataset.task = task
    if env_id is not None:
        cfg.dataset.env_id = int(env_id)
    cfg.dataset.root_dir = resolve_dataset_root(str(cfg.dataset.root_dir), str(cfg.dataset.task))
    cfg.dataset.root_eval_dir = resolve_dataset_root(
        str(cfg.dataset.root_eval_dir),
        str(cfg.dataset.task),
    )
    cfg.dataset.root_test_dir = resolve_dataset_root(
        str(cfg.dataset.root_test_dir),
        str(cfg.dataset.task),
    )
    cfg.dataset.batch_size = int(getattr(cfg.dataset, "batch_size", 32) or 32)
    cfg.dataset.num_workers = 0
    cfg.model.gripper_transition_window = int(max(0, transition_window))
    cfg.model.gripper_transition_weight = max(
        1.0,
        float(getattr(cfg.model, "gripper_transition_weight", 1.0) or 1.0),
    )
    return cfg


def gripper_probabilities(agent: Agent, image, low_dim_state, obs_context):
    if hasattr(agent, "get_action_with_gripper_logits"):
        action, logits = agent.get_action_with_gripper_logits(
            image,
            low_dim_state,
            obs_context=obs_context,
        )
        if logits is not None:
            return torch.sigmoid(logits)
    action = agent.get_action(image, low_dim_state, obs_context=obs_context)
    return ((action[:, 7::8] + 1.0) * 0.5).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.task, args.env_id, args.transition_window)
    cfg.dataset.batch_size = int(args.batch_size)
    cfg.dataset.num_workers = int(args.num_workers)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ImitationDataset(
        config=cfg,
        train=args.split == "train",
        test=args.split == "test",
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    sample = next(iter(loader))
    agent = Agent(
        image_channels=int(sample["image"].shape[1]),
        low_dim_state_dim=int(sample["low_dim_state"].shape[1]),
        action_dim=int(sample["action"].shape[1]),
        image_size=(int(sample["image"].shape[2]), int(sample["image"].shape[3])),
        **get_agent_model_kwargs(cfg),
    )
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=device)
        agent.policy.load_state_dict(state_dict)
    agent.policy.to(device)
    agent.eval()

    global_metrics = MetricAccumulator()
    transition_metrics = MetricAccumulator()
    non_transition_metrics = MetricAccumulator()
    target_flip_counts: list[int] = []
    predicted_flip_counts: list[int] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            image = batch["image"].to(device)
            low_dim_state = batch["low_dim_state"].to(device)
            action = batch["action"].to(device)
            obs_context = move_nested_to_device(batch.get("obs_context"), device)
            prob_open = gripper_probabilities(agent, image, low_dim_state, obs_context)
            target_open = ((action[:, 7::8] + 1.0) * 0.5).clamp(0.0, 1.0)
            transition_mask = batch.get("gripper_transition_mask")
            if transition_mask is None:
                transition_mask = torch.zeros_like(target_open, dtype=torch.bool)
            else:
                transition_mask = transition_mask.to(device) > 0.5

            pred_open = prob_open >= 0.5
            target_binary = target_open >= 0.5
            if target_binary.shape[-1] > 1:
                target_flip_counts.extend(
                    (target_binary[:, 1:] != target_binary[:, :-1]).sum(dim=1).cpu().int().tolist()
                )
                predicted_flip_counts.extend(
                    (pred_open[:, 1:] != pred_open[:, :-1]).sum(dim=1).cpu().int().tolist()
                )

            full_mask = torch.ones_like(target_open, dtype=torch.bool)
            global_metrics.update(prob_open, target_open, full_mask)
            transition_metrics.update(prob_open, target_open, transition_mask)
            non_transition_metrics.update(prob_open, target_open, ~transition_mask)

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "task": str(cfg.dataset.task),
        "env_id": int(cfg.dataset.env_id),
        "split": args.split,
        "dataset_size": len(dataset),
        "transition_window": int(args.transition_window),
        "global": global_metrics.summary(),
        "transition_window_metrics": transition_metrics.summary(),
        "non_transition_metrics": non_transition_metrics.summary(),
        "target_flips_per_sample_mean": (
            float(np.mean(target_flip_counts)) if target_flip_counts else 0.0
        ),
        "predicted_flips_per_sample_mean": (
            float(np.mean(predicted_flip_counts)) if predicted_flip_counts else 0.0
        ),
    }
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
