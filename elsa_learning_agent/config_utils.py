from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf

from elsa_learning_agent.dataset.path_utils import resolve_dataset_root


REPO_ROOT = Path("/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge")
BASE_DATASET_CONFIG_PATH = REPO_ROOT / "dataset_config.yaml"


def _normalize_checkpoint_stem(model_path: str | Path) -> str:
    stem = Path(model_path).stem
    return re.sub(r"_round_\d+$", "", stem)


def infer_checkpoint_config_path(model_path: str | Path) -> Path | None:
    model_path = Path(model_path)
    checkpoint_dir = model_path.parent
    normalized_stem = _normalize_checkpoint_stem(model_path)
    candidates = [
        checkpoint_dir / f"{normalized_stem}.config.yaml",
        checkpoint_dir / f"{normalized_stem}_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_runtime_config(
    config_path: str | Path | None = None,
    *,
    task: str | None = None,
    env_id: int | None = None,
) -> object:
    base_config = OmegaConf.load(BASE_DATASET_CONFIG_PATH)
    if config_path is None:
        config = base_config
    else:
        override_config = OmegaConf.load(str(config_path))
        config = OmegaConf.merge(base_config, override_config)

    if task is not None:
        config.dataset.task = task
    if env_id is not None:
        config.dataset.env_id = int(env_id)

    config.dataset.root_dir = resolve_dataset_root(
        str(config.dataset.root_dir), str(config.dataset.task)
    )
    config.dataset.root_eval_dir = resolve_dataset_root(
        str(config.dataset.root_eval_dir), str(config.dataset.task)
    )
    config.dataset.root_test_dir = resolve_dataset_root(
        str(config.dataset.root_test_dir), str(config.dataset.task)
    )
    return config
