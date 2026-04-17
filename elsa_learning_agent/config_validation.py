from __future__ import annotations

import importlib.util

from elsa_learning_agent.model_registry import (
    get_supported_vision_backbones,
    get_vision_backbone_spec,
)
from elsa_learning_agent.utils import (
    get_action_pipeline_preset,
    get_action_representation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_receding_horizon_execute_steps,
)
from federated_elsa_robotics.fl_method_registry import (
    get_federated_method_preset,
    get_federated_method_spec,
    resolve_prox_mu,
)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def validate_runtime_config(config) -> dict:
    warnings: list[str] = []
    errors: list[str] = []

    vision_backbone = str(getattr(config.model, "vision_backbone", "cnn"))
    spec = get_vision_backbone_spec(vision_backbone)
    if spec.dependency == "timm" and not _module_available("timm"):
        errors.append("vision_backbone requires `timm`, but it is not importable.")
    if spec.dependency == "transformers" and not _module_available("transformers"):
        errors.append("vision_backbone requires `transformers`, but it is not importable.")
    if spec.dependency == "timm + transformers":
        if not _module_available("timm"):
            errors.append("vision_backbone requires `timm`, but it is not importable.")
        if not _module_available("transformers"):
            errors.append("vision_backbone requires `transformers`, but it is not importable.")

    action_pipeline_preset = get_action_pipeline_preset(config)
    action_representation = get_action_representation(config)
    execution_action_interface = get_execution_action_interface(config)
    execution_action_adapter = get_execution_action_adapter(config)

    action_min = list(getattr(config.transform, "action_min", []))
    action_max = list(getattr(config.transform, "action_max", []))
    if len(action_min) != len(action_max):
        errors.append("transform.action_min/action_max lengths do not match.")
    if len(action_min) not in (0, 8):
        warnings.append(
            f"Per-step action bounds dim is {len(action_min)}. Expected 8 for current action pipelines."
        )

    if action_representation.startswith("joint_position") and len(action_min) == 8:
        if action_min[7] != 0.0 or action_max[7] != 1.0:
            warnings.append(
                "Joint-position presets usually expect gripper bounds [0, 1]."
            )

    action_chunk_len = int(getattr(config.dataset, "action_chunk_len", 1) or 1)
    execute_steps = get_receding_horizon_execute_steps(config)
    if execute_steps > action_chunk_len:
        warnings.append(
            f"execute_steps={execute_steps} is larger than action_chunk_len={action_chunk_len}; "
            "runtime will clamp it."
        )

    federated_method_preset = get_federated_method_preset(config)
    federated_method_spec = get_federated_method_spec(config)
    prox_mu = resolve_prox_mu(config)
    if federated_method_spec["local_regularizer"] == "none" and prox_mu > 0.0:
        warnings.append(
            f"prox_mu={prox_mu} is set while federated preset is {federated_method_preset}."
        )

    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))

    return {
        "vision_backbone": vision_backbone,
        "supported_vision_backbones": get_supported_vision_backbones(),
        "action_pipeline_preset": action_pipeline_preset,
        "action_representation": action_representation,
        "execution_action_interface": execution_action_interface,
        "execution_action_adapter": execution_action_adapter,
        "federated_method_preset": federated_method_preset,
        "server_strategy": federated_method_spec["server_strategy"],
        "local_regularizer": federated_method_spec["local_regularizer"],
        "prox_mu": prox_mu,
        "warnings": warnings,
    }
