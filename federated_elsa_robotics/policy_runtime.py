import random

import numpy as np
import torch

from elsa_learning_agent.agent import Agent as LegacyBCAgent
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


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def get_runtime_policy_name(run_config, config):
    if "policy-name" in run_config:
        return run_config["policy-name"]
    return build_policy_kwargs_from_config(config).get("policy_name", DEFAULT_POLICY_NAME)


def build_runtime_agent(
    *,
    policy_name,
    image_size=(128, 128),
    action_dim=8,
    low_dim_state_dim=None,
    config=None,
):
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
