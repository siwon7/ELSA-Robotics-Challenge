import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


REPO_ROOT = Path("/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge")
DATASET_BASE_CANDIDATES = [
    REPO_ROOT / "datasets",
    Path("/mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets"),
]
MAIN_CFG_PATH = REPO_ROOT / "dataset_config.yaml"


def split_root(split: str) -> Path:
    split_name = None
    if split == "training":
        split_name = "training"
    elif split == "eval":
        split_name = "eval"
    elif split == "test":
        split_name = "test"
    if split_name is not None:
        for base in DATASET_BASE_CANDIDATES:
            root = base / split_name
            if root.exists():
                return root
    raise ValueError(f"unknown split={split}")


def load_main_split_cfg(task: str, split: str):
    main_cfg = OmegaConf.load(str(MAIN_CFG_PATH))
    main_cfg.dataset.task = task
    split_name = split_root(split).name
    for base in DATASET_BASE_CANDIDATES:
        task_root = base / split_name / task
        fed_yaml = task_root / f"{task}_fed.yaml"
        fed_json = task_root / f"{task}_fed.json"
        if not fed_yaml.exists() or not fed_json.exists():
            continue
        fed_cfg = OmegaConf.load(str(fed_yaml))
        fed_cfg.dataset = main_cfg.dataset
        fed_cfg.transform = main_cfg.transform
        with open(fed_json, "r", encoding="utf-8") as fh:
            collection_cfg = json.load(fh)
        return fed_cfg, collection_cfg
    raise FileNotFoundError(
        f"Could not find fed config/json for task={task} split={split} "
        f"in any dataset base: {DATASET_BASE_CANDIDATES}"
    )


def serialize_random_state(state):
    if state is None:
        return None
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": np.asarray(keys).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def deserialize_random_state(payload):
    if payload is None:
        return None
    return (
        payload["name"],
        np.asarray(payload["keys"], dtype=np.uint32),
        int(payload["pos"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )
