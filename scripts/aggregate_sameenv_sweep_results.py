#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import yaml


DEFAULT_TASKS = [
    "close_box",
    "slide_block_to_target",
    "insert_onto_square_peg",
    "scoop_with_spatula",
]

CSV_COLUMNS = [
    "task",
    "run_name",
    "env_dir",
    "action_pipeline",
    "vision_backbone",
    "separate_gripper_head",
    "policy_head_type",
    "dino_lora_rank",
    "action_chunk_len",
    "receding_horizon_execute_steps",
    "joint_velocity_servo_gain",
    "joint_velocity_servo_clip",
    "gripper_transition_weight",
    "gripper_transition_window",
    "gripper_eval_mode",
    "family",
    "sr",
    "rmse",
    "result_path",
]


def default_result_root() -> str:
    artifact_root = os.environ.get(
        "ELSA_ARTIFACT_ROOT",
        "/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts",
    )
    return str(pathlib.Path(artifact_root) / "results" / "same_env_suite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate same-env sweep results into CSV or TSV output."
    )
    parser.add_argument(
        "--result-root",
        default=default_result_root(),
        help=(
            "Root directory containing same-env task result subdirectories. "
            "Defaults to ${ELSA_ARTIFACT_ROOT}/results/same_env_suite or the hardcoded fallback."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task subdirectories to scan under --result-root.",
    )
    parser.add_argument(
        "--out",
        default="results/sameenv_sweep_summary.csv",
        help="CSV output path when not using --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the aggregated rows as TSV to stdout instead of writing CSV.",
    )
    return parser.parse_args()


def load_yaml_dict(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def load_json_dict(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def resolve_config_path(result_path: pathlib.Path, result_data: dict) -> pathlib.Path | None:
    run_dir = result_path.parent.parent
    candidates: list[pathlib.Path] = []

    configured_path = result_data.get("resolved_config_path")
    if configured_path:
        candidates.append(pathlib.Path(configured_path))

    candidates.extend(
        [
            run_dir / "resolved_config.yaml",
            result_path.parent / "resolved_config.yaml",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_sr(result_data: dict) -> object:
    for key in ("sr", "success_rate", "mean_per_env_sr"):
        if key in result_data:
            return result_data.get(key)
    return None


def extract_rmse(result_data: dict) -> object:
    offline_seen_env = result_data.get("offline_seen_env")
    if isinstance(offline_seen_env, dict):
        return offline_seen_env.get("rmse")
    return None


def build_row(task: str, run_name: str, result_path: pathlib.Path) -> dict:
    result_data = load_json_dict(result_path)
    config_path = resolve_config_path(result_path, result_data)
    config_data = load_yaml_dict(config_path) if config_path is not None else {}

    dataset_cfg = config_data.get("dataset", {})
    model_cfg = config_data.get("model", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    return {
        "task": task,
        "run_name": run_name,
        "env_dir": result_path.parent.name,
        "action_pipeline": dataset_cfg.get("action_pipeline_preset"),
        "vision_backbone": model_cfg.get("vision_backbone"),
        "separate_gripper_head": model_cfg.get("separate_gripper_head"),
        "policy_head_type": model_cfg.get("policy_head_type"),
        "dino_lora_rank": model_cfg.get("dino_lora_rank"),
        "action_chunk_len": dataset_cfg.get("action_chunk_len", 1),
        "receding_horizon_execute_steps": dataset_cfg.get(
            "receding_horizon_execute_steps",
            1,
        ),
        "joint_velocity_servo_gain": dataset_cfg.get("joint_velocity_servo_gain", ""),
        "joint_velocity_servo_clip": dataset_cfg.get("joint_velocity_servo_clip", ""),
        "gripper_transition_weight": model_cfg.get("gripper_transition_weight", 1.0),
        "gripper_transition_window": model_cfg.get("gripper_transition_window", 0),
        "gripper_eval_mode": dataset_cfg.get("gripper_eval_mode", "threshold"),
        "family": result_path.parents[3].name if len(result_path.parents) > 3 else "",
        "sr": extract_sr(result_data),
        "rmse": extract_rmse(result_data),
        "result_path": str(result_path),
    }


def collect_rows(result_root: pathlib.Path, tasks: list[str]) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        task_dir = result_root / task
        if not task_dir.exists():
            continue
        for run_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            for result_path in sorted(run_dir.glob("env_*/result.json")):
                rows.append(build_row(task, run_dir.name, result_path))
    return rows


def write_csv(rows: list[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(rows: list[dict]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=CSV_COLUMNS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_root = pathlib.Path(args.result_root)
    rows = collect_rows(result_root, list(args.tasks))

    if args.dry_run:
        write_tsv(rows)
        return

    write_csv(rows, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
