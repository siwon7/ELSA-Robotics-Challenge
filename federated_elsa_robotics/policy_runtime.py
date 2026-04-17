import os
import random
from pathlib import Path

import numpy as np
import torch
from flwr.common.config import get_fused_config_from_dir

from elsa_learning_agent.agent import Agent as LegacyBCAgent
from elsa_learning_agent.dataset.compat import get_action_dim
from elsa_learning_agent.agent_forward_kinematics import (
    Agent as FKAgent,
    DEFAULT_POLICY_NAME,
    LOW_DIM_STATE_DIM,
    build_agent_kwargs,
    build_policy_kwargs_from_config,
)


LEGACY_BC_POLICY_NAME = "legacy_bc"
LEGACY_BC_LOW_DIM_STATE_DIM = 8
DEFAULT_SEED = 0
APP_DIR = Path(__file__).resolve().parents[1]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def resolve_run_config(run_config):
    resolved = get_fused_config_from_dir(APP_DIR, {})

    env_overrides = {
        "num-server-rounds": int(os.getenv("ELSA_NUM_SERVER_ROUNDS", resolved["num-server-rounds"])),
        "local-epochs": int(os.getenv("ELSA_LOCAL_EPOCHS", resolved["local-epochs"])),
        "fraction-fit": float(os.getenv("ELSA_FRACTION_FIT", resolved["fraction-fit"])),
        "fraction-eval": float(os.getenv("ELSA_FRACTION_EVAL", resolved["fraction-eval"])),
        "use-wandb": parse_bool(os.getenv("ELSA_USE_WANDB", resolved["use-wandb"])),
        "strategy-name": os.getenv("ELSA_STRATEGY_NAME", resolved["strategy-name"]),
        "dataset-config-path": os.getenv(
            "ELSA_DATASET_CONFIG_PATH", resolved["dataset-config-path"]
        ),
        "dataset-task": os.getenv("ELSA_TASK", resolved["dataset-task"]),
        "policy-name": os.getenv("ELSA_POLICY_NAME", resolved["policy-name"]),
        "server-device": os.getenv("ELSA_SERVER_DEVICE", resolved["server-device"]),
        "seed": int(os.getenv("ELSA_SEED", resolved.get("seed", DEFAULT_SEED))),
        "deterministic-training": parse_bool(
            os.getenv(
                "ELSA_DETERMINISTIC_TRAINING",
                resolved.get("deterministic-training", False),
            )
        ),
        "train-split": float(os.getenv("ELSA_TRAIN_SPLIT", resolved["train-split"])),
        "save-rounds": os.getenv("ELSA_SAVE_ROUNDS", resolved["save-rounds"]),
        "enable-centralized-eval": parse_bool(
            os.getenv(
                "ELSA_ENABLE_CENTRALIZED_EVAL",
                resolved.get("enable-centralized-eval", False),
            )
        ),
        "centralized-eval-simulator": parse_bool(
            os.getenv(
                "ELSA_CENTRALIZED_EVAL_SIMULATOR",
                resolved.get("centralized-eval-simulator", False),
            )
        ),
        "centralized-eval-batch-size": int(
            os.getenv(
                "ELSA_CENTRALIZED_EVAL_BATCH_SIZE",
                resolved.get("centralized-eval-batch-size", 32),
            )
        ),
        "centralized-eval-num-workers": int(
            os.getenv(
                "ELSA_CENTRALIZED_EVAL_NUM_WORKERS",
                resolved.get("centralized-eval-num-workers", 8),
            )
        ),
        "enable-client-local-state": parse_bool(
            os.getenv(
                "ELSA_ENABLE_CLIENT_LOCAL_STATE",
                resolved.get("enable-client-local-state", True),
            )
        ),
    }

    resolved.update(env_overrides)
    resolved.update(dict(run_config))
    return resolved


def get_runtime_policy_name(run_config, config):
    if "policy-name" in run_config:
        return run_config["policy-name"]
    return build_policy_kwargs_from_config(config).get("policy_name", DEFAULT_POLICY_NAME)


def build_runtime_agent(
    *,
    policy_name,
    image_size=(128, 128),
    action_dim=None,
    low_dim_state_dim=None,
    config=None,
):
    if action_dim is None and config is not None:
        action_dim = get_action_dim(config)
    if action_dim is None:
        action_dim = 8
    if policy_name == LEGACY_BC_POLICY_NAME:
        agent = LegacyBCAgent(
            image_channels=3,
            low_dim_state_dim=LEGACY_BC_LOW_DIM_STATE_DIM,
            action_dim=action_dim,
            image_size=image_size,
        )
        agent.legacy_low_dim_state_dim = LEGACY_BC_LOW_DIM_STATE_DIM
        agent.policy_name = LEGACY_BC_POLICY_NAME
        return agent

    kwargs = build_agent_kwargs(
        image_channels=3,
        low_dim_state_dim=low_dim_state_dim or LOW_DIM_STATE_DIM,
        action_dim=action_dim,
        image_size=image_size,
        config=config,
    )
    kwargs["policy_name"] = policy_name
    return FKAgent(**kwargs)


def trim_low_dim_state(agent, low_dim_state):
    legacy_low_dim_state_dim = getattr(agent, "legacy_low_dim_state_dim", None)
    if legacy_low_dim_state_dim is None:
        return low_dim_state
    return torch.cat((low_dim_state[..., :7], low_dim_state[..., -1:]), dim=-1)


def set_global_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
