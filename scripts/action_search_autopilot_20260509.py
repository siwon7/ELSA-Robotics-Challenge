#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


TASKS: dict[str, dict[str, str]] = {
    "close_box": {
        "short": "close",
        "base": "experiments/close_box_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16_gripw4.yaml",
    },
    "scoop_with_spatula": {
        "short": "scoop",
        "base": "experiments/scoop_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16_gripw4.yaml",
    },
    "insert_onto_square_peg": {
        "short": "insert",
        "base": "experiments/insert_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16_gripw4.yaml",
    },
    "slide_block_to_target": {
        "short": "slide",
        "base": "experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jpabs_w2_grid16_gripw4.yaml",
    },
}

TASK_ORDER = [
    "close_box",
    "scoop_with_spatula",
    "insert_onto_square_peg",
    "slide_block_to_target",
]

QUEUE_FILES = [
    ("gpu3-main", "scripts/action_search_manager_20260508_queue.tsv"),
    ("gpu0", "scripts/action_search_manager_20260508_gpu0_queue.tsv"),
    ("gpu1", "scripts/action_search_manager_20260508_gpu1_queue.tsv"),
    ("gpu2", "scripts/action_search_manager_20260508_gpu2_queue.tsv"),
]

ABS_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0]
ABS_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0]
REL_MIN = [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.0]
REL_MAX = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0]
VEL_MIN = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
VEL_MAX = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
TIGHT_VOLUME_BOUNDS = [-0.30, -0.40, 0.75, 0.30, 0.40, 1.10]
BROAD_VOLUME_BOUNDS = [-0.45, -0.55, 0.70, 0.45, 0.55, 1.35]


@dataclass(frozen=True)
class CandidateTemplate:
    suffix: str
    kind: str
    note: str
    epochs: int = 120
    batch_size: int = 16
    eval_episodes: int = 20


CANDIDATES = [
    CandidateTemplate(
        suffix="jpabs_w1_tight_gfilm_eeaux_gripw8_hyst",
        kind="jpabs_w1_tight_hyst",
        note="replay_faithful_abs_hold1_lowdrop_gfilm_eeaux_hysteresis",
        epochs=120,
    ),
    CandidateTemplate(
        suffix="jpabs_c4x2_tight_gfilm_eeaux_gripw8_hyst",
        kind="jpabs_c4x2_tight_hyst",
        note="current_jpabs_chunk_but_lowdrop_gfilm_eeaux_hysteresis",
        epochs=120,
    ),
    CandidateTemplate(
        suffix="jpabs_w1_broad_gripw8_hyst",
        kind="jpabs_w1_broad_hyst",
        note="isolate_abs_hold1_without_tight_volume_or_eeaux",
        epochs=120,
    ),
    CandidateTemplate(
        suffix="jpkey4_servo_tight_gfilm_eeaux_gripw8_hyst",
        kind="jpkey4_servo_tight_hyst",
        note="fixed_keyframe_target_servo_execution_hysteresis",
        epochs=120,
    ),
    CandidateTemplate(
        suffix="jpkey8_servo_tight_gfilm_eeaux_gripw8_hyst",
        kind="jpkey8_servo_tight_hyst",
        note="longer_keyframe_target_servo_execution_hysteresis",
        epochs=120,
    ),
    CandidateTemplate(
        suffix="jprel_w2_servo_tight_gfilm_eeaux_gripw8_hyst",
        kind="jprel_w2_servo_tight_hyst",
        note="relative_joint_delta_with_velocity_servo_safety",
        epochs=100,
    ),
    CandidateTemplate(
        suffix="jprel_c4x2_direct_tight_gfilm_eeaux_gripw8_hyst",
        kind="jprel_c4x2_direct_tight_hyst",
        note="relative_joint_delta_direct_absolute_adapter",
        epochs=100,
    ),
    CandidateTemplate(
        suffix="jv_w1_tight_gfilm_eeaux_gripw8_hyst",
        kind="jv_w1_tight_hyst",
        note="velocity_direct_control_with_volume_and_gripper_head",
        epochs=100,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep action-representation search queues filled with new candidates."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get(
            "ELSA_ARTIFACT_ROOT",
            "/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts",
        ),
    )
    parser.add_argument("--target-sr", type=float, default=0.9)
    parser.add_argument(
        "--min-outstanding-per-queue",
        type=int,
        default=int(os.environ.get("ACTION_AUTOPILOT_MIN_OUTSTANDING_PER_QUEUE", "2")),
    )
    parser.add_argument(
        "--max-add",
        type=int,
        default=int(os.environ.get("ACTION_AUTOPILOT_MAX_ADD", "8")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any], *, dry_run: bool) -> None:
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def set_common_same_env_model(
    cfg: dict[str, Any],
    *,
    tight_volume: bool,
    ee_aux: bool,
    gated_film: bool,
    gripper_loss_weight: float = 6.0,
    gripper_transition_weight: float = 8.0,
    gripper_transition_window: int = 12,
) -> None:
    model = cfg.setdefault("model", {})
    model["low_dim_dropout_prob"] = 0.0
    model["volumedp_volume_bounds"] = (
        TIGHT_VOLUME_BOUNDS if tight_volume else BROAD_VOLUME_BOUNDS
    )
    model["volumedp_grid_shape"] = [16, 16, 16]
    model["separate_gripper_head"] = True
    model["gripper_head_hidden_dim"] = 128
    model["gripper_loss_weight"] = float(gripper_loss_weight)
    model["gripper_transition_window"] = int(gripper_transition_window)
    model["gripper_transition_weight"] = float(gripper_transition_weight)
    if gated_film:
        model["proprio_visual_fusion_mode"] = "gated_global_film"
        model["proprio_visual_fusion_hidden_dim"] = 256
        model["proprio_visual_fusion_scale"] = 1.0
    if ee_aux:
        model["volumedp_emit_voxel_weights"] = True
        model["ee_aux_loss_weight"] = 1.0
        model["ee_aux_sigma"] = 0.05


def set_hysteresis(cfg: dict[str, Any]) -> None:
    dataset = cfg.setdefault("dataset", {})
    dataset["gripper_eval_mode"] = "hysteresis"
    dataset["gripper_open_threshold"] = 0.70
    dataset["gripper_close_threshold"] = 0.30
    dataset["gripper_min_hold_steps"] = 2


def set_transform(cfg: dict[str, Any], action_min: list[float], action_max: list[float]) -> None:
    transform = cfg.setdefault("transform", {})
    transform["action_min"] = list(action_min)
    transform["action_max"] = list(action_max)


def build_candidate_config(
    repo_root: Path,
    task: str,
    template: CandidateTemplate,
) -> tuple[str, dict[str, Any]]:
    task_info = TASKS[task]
    cfg = load_yaml(repo_root / task_info["base"])
    dataset = cfg.setdefault("dataset", {})
    model = cfg.setdefault("model", {})
    dataset["task"] = task
    dataset["env_id"] = 0
    dataset["env_id_test"] = 0

    if template.kind == "jpabs_w1_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_position_direct"
        dataset["action_representation"] = "auto"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_chunk_len"] = 1
        dataset["receding_horizon_execute_steps"] = 1
        model["action_output_activation"] = "identity"
        set_transform(cfg, ABS_MIN, ABS_MAX)
        set_common_same_env_model(cfg, tight_volume=True, ee_aux=True, gated_film=True)
        set_hysteresis(cfg)
    elif template.kind == "jpabs_c4x2_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_position_direct"
        dataset["action_representation"] = "auto"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_chunk_len"] = 4
        dataset["receding_horizon_execute_steps"] = 2
        model["action_output_activation"] = "identity"
        set_transform(cfg, ABS_MIN, ABS_MAX)
        set_common_same_env_model(cfg, tight_volume=True, ee_aux=True, gated_film=True)
        set_hysteresis(cfg)
    elif template.kind == "jpabs_w1_broad_hyst":
        dataset["action_pipeline_preset"] = "joint_position_direct"
        dataset["action_representation"] = "auto"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_chunk_len"] = 1
        dataset["receding_horizon_execute_steps"] = 1
        model["action_output_activation"] = "identity"
        set_transform(cfg, ABS_MIN, ABS_MAX)
        set_common_same_env_model(cfg, tight_volume=False, ee_aux=False, gated_film=False)
        set_hysteresis(cfg)
    elif template.kind == "jpkey4_servo_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_position_to_benchmark_joint_velocity_servo"
        dataset["action_representation"] = "joint_position_keyframe"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_keyframe_horizon"] = 4
        dataset["action_keyframe_selection"] = "fixed_horizon"
        dataset["action_chunk_len"] = 1
        dataset["receding_horizon_execute_steps"] = 1
        dataset["joint_velocity_servo_gain"] = 20.0
        dataset["joint_velocity_servo_clip"] = 1.0
        dataset["joint_velocity_servo_steps"] = 3
        dataset["joint_velocity_servo_tolerance"] = 0.02
        model["action_output_activation"] = "identity"
        set_transform(cfg, ABS_MIN, ABS_MAX)
        set_common_same_env_model(cfg, tight_volume=True, ee_aux=True, gated_film=True)
        set_hysteresis(cfg)
    elif template.kind == "jpkey8_servo_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_position_to_benchmark_joint_velocity_servo"
        dataset["action_representation"] = "joint_position_keyframe"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_keyframe_horizon"] = 8
        dataset["action_keyframe_selection"] = "fixed_horizon"
        dataset["action_chunk_len"] = 1
        dataset["receding_horizon_execute_steps"] = 1
        dataset["joint_velocity_servo_gain"] = 20.0
        dataset["joint_velocity_servo_clip"] = 1.0
        dataset["joint_velocity_servo_steps"] = 3
        dataset["joint_velocity_servo_tolerance"] = 0.02
        model["action_output_activation"] = "identity"
        set_transform(cfg, ABS_MIN, ABS_MAX)
        set_common_same_env_model(cfg, tight_volume=True, ee_aux=True, gated_film=True)
        set_hysteresis(cfg)
    elif template.kind == "jprel_w2_servo_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_position_relative_to_benchmark_joint_velocity_servo"
        dataset["action_representation"] = "auto"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_chunk_len"] = 2
        dataset["receding_horizon_execute_steps"] = 2
        dataset["joint_velocity_servo_gain"] = 20.0
        dataset["joint_velocity_servo_clip"] = 1.0
        dataset["joint_velocity_servo_steps"] = 2
        dataset["joint_velocity_servo_tolerance"] = 0.01
        model["action_output_activation"] = "identity"
        set_transform(cfg, REL_MIN, REL_MAX)
        set_common_same_env_model(cfg, tight_volume=True, ee_aux=True, gated_film=True)
        set_hysteresis(cfg)
    elif template.kind == "jprel_c4x2_direct_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_position_relative_direct"
        dataset["action_representation"] = "auto"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_chunk_len"] = 4
        dataset["receding_horizon_execute_steps"] = 2
        model["action_output_activation"] = "identity"
        set_transform(cfg, REL_MIN, REL_MAX)
        set_common_same_env_model(cfg, tight_volume=True, ee_aux=True, gated_film=True)
        set_hysteresis(cfg)
    elif template.kind == "jv_w1_tight_hyst":
        dataset["action_pipeline_preset"] = "joint_velocity_direct"
        dataset["action_representation"] = "joint_velocity"
        dataset["execution_action_interface"] = "auto"
        dataset["execution_action_adapter"] = "auto"
        dataset["action_chunk_len"] = 1
        dataset["receding_horizon_execute_steps"] = 1
        model["action_output_activation"] = "tanh"
        set_transform(cfg, VEL_MIN, VEL_MAX)
        set_common_same_env_model(
            cfg,
            tight_volume=True,
            ee_aux=True,
            gated_film=True,
            gripper_loss_weight=4.0,
            gripper_transition_weight=8.0,
            gripper_transition_window=12,
        )
        set_hysteresis(cfg)
    else:
        raise ValueError(f"Unknown candidate kind: {template.kind}")

    rel_path = (
        Path("experiments/action_search_autopilot_20260509")
        / f"{task_info['short']}_{template.suffix}.yaml"
    )
    return str(rel_path), cfg


def read_queue(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for values in reader:
            if not values or values[0].startswith("#"):
                continue
            if len(values) < 9:
                continue
            rows.append(
                {
                    "enabled": values[0],
                    "priority": values[1],
                    "task": values[2],
                    "config": values[3],
                    "run_name": values[4],
                    "epochs": values[5],
                    "batch_size": values[6],
                    "eval_episodes": values[7],
                    "note": values[8],
                }
            )
    return rows


def result_paths(result_root: Path, task: str, run_name: str) -> list[str]:
    return sorted(glob.glob(str(result_root / task / run_name / "*" / "result.json")))


def result_exists(result_root: Path, task: str, run_name: str) -> bool:
    return bool(result_paths(result_root, task, run_name))


def collect_known_runs(repo_root: Path, result_root: Path) -> set[str]:
    known: set[str] = set()
    for _label, rel_queue in QUEUE_FILES:
        for row in read_queue(repo_root / rel_queue):
            known.add(row["run_name"])
    for path in result_root.glob("*/*/*/result.json"):
        known.add(path.parents[1].name)
    return known


def queue_outstanding(repo_root: Path, result_root: Path, rel_queue: str) -> int:
    outstanding = 0
    for row in read_queue(repo_root / rel_queue):
        if row["enabled"] != "1":
            continue
        if not result_exists(result_root, row["task"], row["run_name"]):
            outstanding += 1
    return outstanding


def best_results(result_root: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {
        task: {"sr": -1.0, "run_name": "none", "path": ""}
        for task in TASK_ORDER
    }
    for path in result_root.glob("*/*/*/result.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        task = str(payload.get("task") or path.parents[2].name)
        if task not in best:
            continue
        sr = payload.get("mean_per_env_sr", payload.get("sr", -1.0))
        try:
            sr_f = float(sr)
        except Exception:
            continue
        if sr_f > float(best[task]["sr"]):
            best[task] = {
                "sr": sr_f,
                "run_name": path.parents[1].name,
                "path": str(path),
            }
    return best


def candidate_stream(best: dict[str, dict[str, Any]], target_sr: float):
    below_target = [
        task for task in TASK_ORDER if float(best.get(task, {}).get("sr", -1.0)) < target_sr
    ]
    for template in CANDIDATES:
        for task in below_target:
            yield task, template


def append_queue_row(
    repo_root: Path,
    queue_rel: str,
    row: list[str],
    *,
    dry_run: bool,
) -> None:
    queue_path = repo_root / queue_rel
    if dry_run:
        return
    if not queue_path.exists():
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(
            "# enabled\tpriority\ttask\tconfig\trun_name\tepochs\tbatch_size\teval_episodes\tnote\n",
            encoding="utf-8",
        )
    with queue_path.open("a", encoding="utf-8") as fh:
        fh.write("\t".join(row) + "\n")


def choose_queue(
    queue_depth: dict[str, int],
    min_outstanding: int,
) -> str | None:
    available = [
        (count, rel_queue)
        for rel_queue, count in queue_depth.items()
        if count < min_outstanding
    ]
    if not available:
        return None
    available.sort(key=lambda item: (item[0], item[1]))
    return available[0][1]


def write_status(
    status_path: Path,
    *,
    best: dict[str, dict[str, Any]],
    queue_depth: dict[str, int],
    added: list[dict[str, str]],
    target_sr: float,
    dry_run: bool,
) -> None:
    lines = [
        "# Action Search Autopilot Status",
        "",
        f"- time: {now()}",
        f"- target_sr: {target_sr}",
        f"- dry_run: {dry_run}",
        "",
        "## Best manager results",
    ]
    for task in TASK_ORDER:
        item = best[task]
        lines.append(
            f"- {task}: sr={item['sr']} run={item['run_name']}"
        )
    lines.extend(["", "## Queue outstanding"])
    for label, rel_queue in QUEUE_FILES:
        lines.append(f"- {label}: {queue_depth.get(rel_queue, 0)} outstanding ({rel_queue})")
    lines.extend(["", "## Added this pass"])
    if not added:
        lines.append("- none")
    else:
        for item in added:
            lines.append(
                "- "
                f"queue={item['queue']} task={item['task']} run={item['run_name']} "
                f"cfg={item['config']} note={item['note']}"
            )
    lines.extend(
        [
            "",
            "## Strategy",
            "- Prefer replay-faithful joint-position absolute hold-1 because replay ceiling hit 1.0 on insert/scoop.",
            "- Add same-env low-dropout, proprio gated-film, VolumeDP tight bounds, EE auxiliary supervision, and gripper hysteresis.",
            "- Fall back to fixed keyframe servo and relative/velocity controls if direct absolute keeps failing.",
        ]
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    result_root = artifact_root / "results" / "action_search_manager_20260508"
    log_root = artifact_root / "logs" / "action_search_autopilot_20260509"
    lock_path = log_root / "autopilot.lock"
    status_path = result_root / "AUTOPILOT_STATUS.md"

    log_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[{now()}] another autopilot pass is active")
            return 0

        best = best_results(result_root)
        known_runs = collect_known_runs(repo_root, result_root)
        queue_depth = {
            rel_queue: queue_outstanding(repo_root, result_root, rel_queue)
            for _label, rel_queue in QUEUE_FILES
        }
        added: list[dict[str, str]] = []
        next_priority = 1000

        for task, template in candidate_stream(best, args.target_sr):
            if len(added) >= args.max_add:
                break
            queue_rel = choose_queue(queue_depth, args.min_outstanding_per_queue)
            if queue_rel is None:
                break
            short = TASKS[task]["short"]
            run_name = f"{short}_{template.suffix}_auto20260509_s0"
            if run_name in known_runs:
                continue
            config_rel, cfg = build_candidate_config(repo_root, task, template)
            dump_yaml(repo_root / config_rel, cfg, dry_run=args.dry_run)
            append_queue_row(
                repo_root,
                queue_rel,
                [
                    "1",
                    str(next_priority),
                    task,
                    config_rel,
                    run_name,
                    str(template.epochs),
                    str(template.batch_size),
                    str(template.eval_episodes),
                    template.note,
                ],
                dry_run=args.dry_run,
            )
            added.append(
                {
                    "queue": queue_rel,
                    "task": task,
                    "run_name": run_name,
                    "config": config_rel,
                    "note": template.note,
                }
            )
            known_runs.add(run_name)
            queue_depth[queue_rel] = queue_depth.get(queue_rel, 0) + 1
            next_priority += 10

        write_status(
            status_path,
            best=best,
            queue_depth=queue_depth,
            added=added,
            target_sr=args.target_sr,
            dry_run=args.dry_run,
        )
        print(json.dumps({"time": now(), "added": added, "queue_depth": queue_depth}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
