from __future__ import annotations

from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path("/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge")
DATASET_BASE_CANDIDATES = [
    REPO_ROOT / "datasets",
    Path("/mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets"),
]


def resolve_dataset_root(path_value: str, task: str) -> str:
    path = Path(path_value)
    if (path / task).exists():
        return str(path)
    split_name = path.name
    if split_name in ("training", "eval", "test"):
        for base in DATASET_BASE_CANDIDATES:
            candidate = base / split_name
            if (candidate / task).exists():
                return str(candidate)
    return str(path)


@lru_cache(maxsize=16)
def available_env_ids(root_dir: str, task: str) -> tuple[int, ...]:
    root_path = Path(root_dir)
    candidate_task_roots: list[Path] = [root_path / task]
    split_name = root_path.name
    if split_name in ("training", "eval", "test"):
        for base in DATASET_BASE_CANDIDATES:
            candidate_task_roots.append(base / split_name / task)

    for task_root in candidate_task_roots:
        env_ids: list[int] = []
        for env_dir in sorted(task_root.glob("env_*")):
            try:
                env_id = int(env_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if (env_dir / "episodes_observations.pkl.gz").exists():
                env_ids.append(env_id)
        if env_ids:
            return tuple(env_ids)
    raise FileNotFoundError(f"No dataset shards found under {candidate_task_roots}")


def resolve_existing_env_id(root_dir: str, task: str, env_id: int) -> int:
    env_ids = available_env_ids(root_dir, task)
    if env_id in env_ids:
        return int(env_id)
    return int(env_ids[env_id % len(env_ids)])
