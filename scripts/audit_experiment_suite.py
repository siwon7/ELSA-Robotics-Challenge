#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.config_validation import validate_runtime_config
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.dataset.path_utils import (
    resolve_dataset_root,
    resolve_existing_env_id,
)
from federated_elsa_robotics.task import infer_action_dim


@dataclass
class AuditResult:
    config: str
    status: str
    task: str | None = None
    env_id: int | None = None
    dataset_size: int | None = None
    action_shape: list[int] | None = None
    predicted_shape: list[int] | None = None
    validation_summary: dict | None = None
    error: str | None = None
    note: str | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit all experiment YAMLs against current code paths."
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Directory containing experiment YAMLs.",
    )
    parser.add_argument(
        "--output",
        default="results/experiment_audit/latest_report.json",
        help="Path to write the JSON audit report.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Probe datasets with normalized actions enabled.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of experiment files for debugging.",
    )
    return parser


def load_merged_config(base_config_path: Path, experiment_path: Path):
    base_config = OmegaConf.load(base_config_path)
    override_config = OmegaConf.load(experiment_path)
    config = OmegaConf.merge(base_config, override_config)
    task = str(getattr(config.dataset, "task", "") or "")
    if not task:
        if experiment_path.name.startswith("slide_block_to_target"):
            task = "slide_block_to_target"
        elif experiment_path.name.startswith("close_box"):
            task = "close_box"
        elif experiment_path.name.startswith("insert_onto_square_peg"):
            task = "insert_onto_square_peg"
        elif experiment_path.name.startswith("scoop_with_spatula"):
            task = "scoop_with_spatula"
    config.dataset.task = task
    config.dataset.root_dir = resolve_dataset_root(str(config.dataset.root_dir), task)
    config.dataset.root_eval_dir = resolve_dataset_root(str(config.dataset.root_eval_dir), task)
    config.dataset.root_test_dir = resolve_dataset_root(str(config.dataset.root_test_dir), task)
    config.dataset.env_id = resolve_existing_env_id(
        str(config.dataset.root_dir),
        task,
        int(getattr(config.dataset, "env_id", 0) or 0),
    )
    return config


def is_template_experiment(experiment_path: Path) -> bool:
    return experiment_path.name.endswith("_template.yaml")


def audit_experiment(base_config_path: Path, experiment_path: Path, normalize: bool) -> AuditResult:
    if is_template_experiment(experiment_path):
        return AuditResult(
            config=str(experiment_path),
            status="skipped",
            note="Template config; dataset.task and concrete shard are intentionally unspecified.",
        )
    try:
        config = load_merged_config(base_config_path, experiment_path)
        validation_summary = validate_runtime_config(config)
        config.dataset.action_dim = infer_action_dim(config)
        dataset = ImitationDataset(config=config, train=True, normalize=normalize)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after merge.")
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
        return AuditResult(
            config=str(experiment_path),
            status="ok",
            task=str(config.dataset.task),
            env_id=int(config.dataset.env_id),
            dataset_size=len(dataset),
            action_shape=list(action.shape),
            predicted_shape=list(predicted.shape),
            validation_summary=validation_summary,
        )
    except Exception as exc:  # noqa: BLE001
        task = None
        env_id = None
        try:
            config = load_merged_config(base_config_path, experiment_path)
            task = str(config.dataset.task)
            env_id = int(config.dataset.env_id)
        except Exception:  # noqa: BLE001
            pass
        return AuditResult(
            config=str(experiment_path),
            status="error",
            task=task,
            env_id=env_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def main():
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = repo_root / "dataset_config.yaml"
    experiments_dir = repo_root / args.experiments_dir
    experiment_paths = sorted(experiments_dir.glob("*.yaml"))
    if args.limit > 0:
        experiment_paths = experiment_paths[: args.limit]

    results = [
        asdict(audit_experiment(base_config_path, path, normalize=args.normalize))
        for path in experiment_paths
    ]

    summary = {
        "num_experiments": len(results),
        "num_ok": sum(r["status"] == "ok" for r in results),
        "num_skipped": sum(r["status"] == "skipped" for r in results),
        "num_error": sum(r["status"] == "error" for r in results),
        "results": results,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
