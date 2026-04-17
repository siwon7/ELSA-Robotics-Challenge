"""Preflight dataset integrity checks for RLBench env shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from colosseum.rlbench.datacontainer import DataContainer


TASKS = [
    "slide_block_to_target",
    "close_box",
    "insert_onto_square_peg",
    "scoop_with_spatula",
]


def iter_env_paths(root: Path, task: str):
    task_root = root / task
    if not task_root.exists():
        return
    for env_dir in sorted(task_root.glob("env_*")):
        if env_dir.is_dir():
            yield env_dir


def check_env(env_dir: Path, retries: int) -> tuple[bool, str]:
    data_path = env_dir / "episodes_observations.pkl.gz"
    if not data_path.exists():
        return False, "missing episodes_observations.pkl.gz"

    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            DataContainer().load(str(data_path))
            return True, f"ok after {attempt} attempt(s)"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    return False, last_error


def main():
    parser = argparse.ArgumentParser(description="Check ELSA dataset shards before FL runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./datasets/training"),
        help="Dataset split root to inspect, e.g. ./datasets/training or ./datasets/eval",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task to inspect. Pass multiple times, or omit to inspect all four tasks.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="How many times to reopen a shard before marking it as failed.",
    )
    args = parser.parse_args()

    tasks = args.tasks or TASKS
    bad_envs: list[tuple[str, str, str]] = []

    for task in tasks:
        task_root = args.root / task
        if not task_root.exists():
            print(f"[skip] {task}: {task_root} does not exist")
            continue

        print(f"[task] {task}")
        for env_dir in iter_env_paths(args.root, task):
            ok, detail = check_env(env_dir, retries=args.retries)
            status = "ok" if ok else "fail"
            print(f"  [{status}] {env_dir.name}: {detail}")
            if not ok:
                bad_envs.append((task, env_dir.name, detail))

    if bad_envs:
        print("\n[summary] failing env shards")
        for task, env_name, detail in bad_envs:
            print(f"- {task}/{env_name}: {detail}")
        raise SystemExit(1)

    print("\n[summary] all checked env shards loaded successfully")


if __name__ == "__main__":
    main()
