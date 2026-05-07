#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


TASKS = [
    "close_box",
    "slide_block_to_target",
    "insert_onto_square_peg",
    "scoop_with_spatula",
]


def default_artifact_root() -> Path:
    return Path(
        os.environ.get(
            "ELSA_ARTIFACT_ROOT",
            "/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect candidate checkpoints/results for TGAC E1-E3 eval-only experiments."
    )
    parser.add_argument("--artifact-root", type=Path, default=default_artifact_root())
    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "fill4_power_moved_20260507",
            "action_ablation_20260504",
            "jpabs_seedsweep_20260504",
            "overnight_queue",
            "recommended_followups_20260504",
            "same_env_suite",
        ],
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_yaml(path: Path | None) -> dict:
    if path is None or yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def to_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_sr(data: dict) -> float | None:
    for key in ("sr", "success_rate", "mean_per_env_sr"):
        if key in data:
            return to_float(data.get(key))
    online = data.get("online_seen_env")
    if isinstance(online, dict):
        return to_float(online.get("mean_reward"))
    return None


def resolve_config_path(result_path: Path, result: dict) -> Path | None:
    configured = result.get("resolved_config_path")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            result_path.parent / "resolved_config.yaml",
            result_path.parent.parent / "resolved_config.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_checkpoint_path(artifact_root: Path, family: str, task: str, run_name: str, env_dir: str, result: dict) -> str | None:
    configured = result.get("checkpoint_path")
    if configured and Path(configured).exists():
        return configured
    ckpt_root = artifact_root / "model_checkpoints" / family / task / run_name
    env_label = env_dir if env_dir.startswith("env_") else "env_000"
    candidate = ckpt_root / f"{env_label}.pth"
    if candidate.exists():
        return str(candidate)
    return configured if configured else None


def collect_family(artifact_root: Path, family: str, tasks: list[str]) -> list[dict]:
    family_root = artifact_root / "results" / family
    if not family_root.exists():
        return []
    rows: list[dict] = []
    for task in tasks:
        task_root = family_root / task
        if not task_root.exists():
            continue
        for result_path in sorted(task_root.glob("*/env_*/result.json")):
            result = load_json(result_path)
            sr = extract_sr(result)
            run_name = result_path.parents[1].name
            env_dir = result_path.parent.name
            config_path = resolve_config_path(result_path, result)
            config = load_yaml(config_path)
            dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
            model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
            if not isinstance(dataset_cfg, dict):
                dataset_cfg = {}
            if not isinstance(model_cfg, dict):
                model_cfg = {}
            rows.append(
                {
                    "family": family,
                    "task": task,
                    "run_name": run_name,
                    "env_dir": env_dir,
                    "sr": sr,
                    "result_path": str(result_path),
                    "checkpoint_path": infer_checkpoint_path(
                        artifact_root,
                        family,
                        task,
                        run_name,
                        env_dir,
                        result,
                    ),
                    "resolved_config_path": str(config_path) if config_path else None,
                    "action_pipeline_preset": dataset_cfg.get("action_pipeline_preset"),
                    "action_chunk_len": dataset_cfg.get("action_chunk_len"),
                    "receding_horizon_execute_steps": dataset_cfg.get("receding_horizon_execute_steps"),
                    "execution_action_interface": dataset_cfg.get("execution_action_interface"),
                    "execution_action_adapter": dataset_cfg.get("execution_action_adapter"),
                    "vision_backbone": model_cfg.get("vision_backbone"),
                    "policy_head_type": model_cfg.get("policy_head_type"),
                    "separate_gripper_head": model_cfg.get("separate_gripper_head"),
                    "gripper_loss_weight": model_cfg.get("gripper_loss_weight"),
                }
            )
    return rows


def sort_key(row: dict) -> tuple[int, float, str]:
    sr = row.get("sr")
    if sr is None:
        return (1, 0.0, row["run_name"])
    return (0, -float(sr), row["run_name"])


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for family in args.families:
        rows.extend(collect_family(args.artifact_root, family, TASKS))

    by_task = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        task_rows.sort(key=sort_key)
        by_task[task] = task_rows[: max(1, int(args.top_k))]

    payload = {
        "artifact_root": str(args.artifact_root),
        "families": args.families,
        "top_k": int(args.top_k),
        "by_task": by_task,
        "notes": [
            "Use these candidates for TGAC E1-E3 only after checking mixed-era comparability.",
            "Prefer current fill4/action_ablation checkpoints when available; archived same_env_suite rows are useful counterexamples but may not be matched reruns."
        ],
    }
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
