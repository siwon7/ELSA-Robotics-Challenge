#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib

import yaml


def default_result_root() -> str:
    artifact_root = os.environ.get(
        "ELSA_ARTIFACT_ROOT",
        "/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts",
    )
    return str(pathlib.Path(artifact_root) / "results" / "same_env_suite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan same-env results for run names containing one or more wave tags "
            "and write a markdown summary."
        )
    )
    parser.add_argument(
        "--waves",
        nargs="+",
        required=True,
        help="One or more substring tags to match against run names, for example: phase1A",
    )
    parser.add_argument(
        "--result-root",
        default=default_result_root(),
        help=(
            "Root directory containing same-env task subdirectories. Defaults to "
            "${ELSA_ARTIFACT_ROOT}/results/same_env_suite or the hardcoded fallback."
        ),
    )
    parser.add_argument(
        "--out",
        default="results/wave_summary.md",
        help="Markdown output path. Defaults to results/wave_summary.md.",
    )
    return parser.parse_args()


def load_json_dict(path: pathlib.Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_yaml_dict(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def to_float(value: object) -> float | None:
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


def extract_sr(result_data: dict) -> float | None:
    for key in ("sr", "success_rate", "mean_per_env_sr"):
        if key in result_data:
            return to_float(result_data.get(key))
    return None


def extract_rmse(result_data: dict) -> float | None:
    offline_seen_env = result_data.get("offline_seen_env")
    if isinstance(offline_seen_env, dict):
        return to_float(offline_seen_env.get("rmse"))
    return None


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def format_servo(gain: object, clip: object) -> str:
    gain_value = to_float(gain)
    clip_value = to_float(clip)
    if gain_value is None and clip_value is None:
        return "-"
    return f"{format_metric(gain_value)}/{format_metric(clip_value)}"


def format_splitgripper(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "-"


def format_chunk(chunk_len: object, exec_steps: object) -> str:
    chunk_value = int(chunk_len) if isinstance(chunk_len, int) else 1
    exec_value = int(exec_steps) if isinstance(exec_steps, int) else 1
    return f"{chunk_value}/{exec_value}"


def escape_cell(text: object) -> str:
    return str(text).replace("|", "\\|")


def row_sort_key(row: dict) -> tuple[bool, float, str]:
    sr = row["sr_value"]
    sr_key = -sr if sr is not None else 0.0
    return (sr is None, sr_key, row["run"])


def build_row(task: str, run_dir: pathlib.Path, result_path: pathlib.Path) -> dict:
    result_data = load_json_dict(result_path)
    config_data = load_yaml_dict(run_dir / "resolved_config.yaml")

    dataset_cfg = config_data.get("dataset", {})
    model_cfg = config_data.get("model", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    sr = extract_sr(result_data)
    rmse = extract_rmse(result_data)

    return {
        "task": task,
        "run": f"{run_dir.name}/{result_path.parent.name}",
        "action": dataset_cfg.get("action_pipeline_preset", "-"),
        "servo": format_servo(
            dataset_cfg.get("joint_velocity_servo_gain"),
            dataset_cfg.get("joint_velocity_servo_clip"),
        ),
        "splitgripper": format_splitgripper(model_cfg.get("separate_gripper_head")),
        "chunk": format_chunk(
            dataset_cfg.get("action_chunk_len", 1),
            dataset_cfg.get("receding_horizon_execute_steps", 1),
        ),
        "sr": format_metric(sr),
        "rmse": format_metric(rmse),
        "sr_value": sr,
    }


def collect_rows(result_root: pathlib.Path, waves: list[str]) -> dict[str, list[dict]]:
    rows_by_task: dict[str, list[dict]] = {}
    if not result_root.exists():
        return rows_by_task

    task_dirs = sorted(path for path in result_root.iterdir() if path.is_dir())
    for task_dir in task_dirs:
        task_rows: list[dict] = []
        run_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir())
        for run_dir in run_dirs:
            if not any(wave in run_dir.name for wave in waves):
                continue
            for result_path in sorted(run_dir.glob("env_*/result.json")):
                task_rows.append(build_row(task_dir.name, run_dir, result_path))
        if task_rows:
            task_rows.sort(key=row_sort_key)
            rows_by_task[task_dir.name] = task_rows
    return rows_by_task


def render_markdown(waves: list[str], rows_by_task: dict[str, list[dict]]) -> str:
    lines = [f"# Wave summary: {', '.join(waves)}", ""]
    for task in sorted(rows_by_task):
        lines.extend(
            [
                f"## {task}",
                "",
                "| Run | Action | Servo (gain/clip) | Splitgripper | Chunk | SR | RMSE |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows_by_task[task]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_cell(row["run"]),
                        escape_cell(row["action"]),
                        escape_cell(row["servo"]),
                        escape_cell(row["splitgripper"]),
                        escape_cell(row["chunk"]),
                        escape_cell(row["sr"]),
                        escape_cell(row["rmse"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    rows_by_task = collect_rows(pathlib.Path(args.result_root), list(args.waves))
    markdown = render_markdown(list(args.waves), rows_by_task)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
