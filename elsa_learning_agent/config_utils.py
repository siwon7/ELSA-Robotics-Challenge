from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf

from elsa_learning_agent.dataset.path_utils import resolve_dataset_root
from elsa_learning_agent.utils import get_action_output_activation


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


def get_agent_model_kwargs(config) -> dict:
    return {
        "vision_backbone": str(getattr(config.model, "vision_backbone", "cnn")),
        "projector_dim": int(getattr(config.model, "projector_dim", 256)),
        "action_output_activation": get_action_output_activation(config),
        "normalize_branch_embeddings": bool(
            getattr(config.model, "normalize_branch_embeddings", False)
        ),
        "low_dim_dropout_prob": float(
            getattr(config.model, "low_dim_dropout_prob", 0.0) or 0.0
        ),
        "use_adaln_head": bool(getattr(config.model, "use_adaln_head", False)),
        "adaln_hidden_dim": int(getattr(config.model, "adaln_hidden_dim", 256) or 256),
        "adaln_conditioning_mode": str(
            getattr(config.model, "adaln_conditioning_mode", "hybrid") or "hybrid"
        ),
        "use_dino_lora": bool(getattr(config.model, "use_dino_lora", False)),
        "dino_lora_rank": int(getattr(config.model, "dino_lora_rank", 8) or 8),
        "dino_lora_alpha": float(getattr(config.model, "dino_lora_alpha", 16.0) or 16.0),
        "dino_lora_dropout": float(
            getattr(config.model, "dino_lora_dropout", 0.0) or 0.0
        ),
        "dino_lora_num_blocks": int(
            getattr(config.model, "dino_lora_num_blocks", 0) or 0
        ),
        "dino_lora_target_modules": str(
            getattr(config.model, "dino_lora_target_modules", "qkv,proj") or "qkv,proj"
        ),
        "policy_head_type": str(getattr(config.model, "policy_head_type", "mlp") or "mlp"),
        "diffusion_num_steps": int(
            getattr(config.model, "diffusion_num_steps", 20) or 20
        ),
        "diffusion_hidden_dim": int(
            getattr(config.model, "diffusion_hidden_dim", 512) or 512
        ),
        "diffusion_timestep_dim": int(
            getattr(config.model, "diffusion_timestep_dim", 128) or 128
        ),
    }
