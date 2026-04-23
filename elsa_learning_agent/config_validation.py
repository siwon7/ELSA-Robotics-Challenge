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
    use_adaln_head = bool(getattr(config.model, "use_adaln_head", False))
    adaln_hidden_dim = int(getattr(config.model, "adaln_hidden_dim", 256) or 256)
    adaln_conditioning_mode = str(
        getattr(config.model, "adaln_conditioning_mode", "hybrid") or "hybrid"
    )
    use_dino_lora = bool(getattr(config.model, "use_dino_lora", False))
    dino_lora_rank = int(getattr(config.model, "dino_lora_rank", 8) or 8)
    dino_lora_alpha = float(getattr(config.model, "dino_lora_alpha", 16.0) or 16.0)
    dino_lora_dropout = float(getattr(config.model, "dino_lora_dropout", 0.0) or 0.0)
    dino_lora_num_blocks = int(getattr(config.model, "dino_lora_num_blocks", 0) or 0)
    dino_lora_target_modules = str(
        getattr(config.model, "dino_lora_target_modules", "qkv,proj") or "qkv,proj"
    )
    policy_head_type = str(getattr(config.model, "policy_head_type", "mlp") or "mlp")
    diffusion_num_steps = int(getattr(config.model, "diffusion_num_steps", 20) or 20)
    diffusion_hidden_dim = int(getattr(config.model, "diffusion_hidden_dim", 512) or 512)
    diffusion_timestep_dim = int(
        getattr(config.model, "diffusion_timestep_dim", 128) or 128
    )
    volumedp_volume_bounds = list(
        getattr(
            config.model,
            "volumedp_volume_bounds",
            [-0.45, -0.55, 0.70, 0.45, 0.55, 1.35],
        )
        or [-0.45, -0.55, 0.70, 0.45, 0.55, 1.35]
    )
    volumedp_grid_shape = list(
        getattr(config.model, "volumedp_grid_shape", [8, 8, 8]) or [8, 8, 8]
    )
    volumedp_num_spatial_tokens = int(
        getattr(config.model, "volumedp_num_spatial_tokens", 32) or 32
    )
    volumedp_decoder_layers = int(
        getattr(config.model, "volumedp_decoder_layers", 2) or 2
    )
    volumedp_decoder_heads = int(
        getattr(config.model, "volumedp_decoder_heads", 4) or 4
    )
    volumedp_action_token_dim = int(
        getattr(config.model, "volumedp_action_token_dim", 8) or 8
    )
    proprio_visual_fusion_mode = str(
        getattr(config.model, "proprio_visual_fusion_mode", "token") or "token"
    )
    proprio_visual_fusion_hidden_dim = int(
        getattr(config.model, "proprio_visual_fusion_hidden_dim", 256) or 256
    )
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
    if adaln_hidden_dim <= 0:
        errors.append("model.adaln_hidden_dim must be positive.")
    if adaln_conditioning_mode not in {"proprio", "image", "hybrid"}:
        errors.append(
            "model.adaln_conditioning_mode must be one of "
            "['proprio', 'image', 'hybrid']."
        )
    if policy_head_type not in {"mlp", "diffusion"}:
        errors.append("model.policy_head_type must be one of ['mlp', 'diffusion'].")
    if proprio_visual_fusion_mode not in {
        "none",
        "token",
        "global_film",
        "token_film",
        "global_token_film",
    }:
        errors.append(
            "model.proprio_visual_fusion_mode must be one of "
            "['none', 'token', 'global_film', 'token_film', 'global_token_film']."
        )
    if proprio_visual_fusion_hidden_dim <= 0:
        errors.append("model.proprio_visual_fusion_hidden_dim must be positive.")
    if diffusion_num_steps <= 1:
        errors.append("model.diffusion_num_steps must be greater than 1.")
    if diffusion_hidden_dim <= 0:
        errors.append("model.diffusion_hidden_dim must be positive.")
    if diffusion_timestep_dim <= 0:
        errors.append("model.diffusion_timestep_dim must be positive.")
    if vision_backbone.startswith("volumedp_lite"):
        if policy_head_type != "diffusion":
            errors.append("VolumeDP-lite vision_backbone requires model.policy_head_type='diffusion'.")
        if len(volumedp_volume_bounds) != 6:
            errors.append("model.volumedp_volume_bounds must contain 6 floats.")
        if len(volumedp_grid_shape) != 3:
            errors.append("model.volumedp_grid_shape must contain 3 ints.")
        elif any(int(value) <= 0 for value in volumedp_grid_shape):
            errors.append("model.volumedp_grid_shape values must be positive.")
        if volumedp_num_spatial_tokens <= 0:
            errors.append("model.volumedp_num_spatial_tokens must be positive.")
        if volumedp_decoder_layers <= 0:
            errors.append("model.volumedp_decoder_layers must be positive.")
        if volumedp_decoder_heads <= 0:
            errors.append("model.volumedp_decoder_heads must be positive.")
        if volumedp_action_token_dim <= 0:
            errors.append("model.volumedp_action_token_dim must be positive.")
    if use_dino_lora:
        if "dinov3" not in vision_backbone:
            errors.append("model.use_dino_lora=true requires a DINOv3 vision_backbone.")
        if dino_lora_rank <= 0:
            errors.append("model.dino_lora_rank must be positive.")
        if dino_lora_alpha <= 0:
            errors.append("model.dino_lora_alpha must be positive.")
        if dino_lora_dropout < 0.0:
            errors.append("model.dino_lora_dropout must be non-negative.")
        if dino_lora_num_blocks <= 0:
            errors.append("model.dino_lora_num_blocks must be positive when LoRA is enabled.")
        valid_targets = {"qkv", "proj"}
        targets = {token.strip() for token in dino_lora_target_modules.split(",") if token.strip()}
        if not targets:
            errors.append("model.dino_lora_target_modules must not be empty.")
        elif not targets.issubset(valid_targets):
            errors.append(
                "model.dino_lora_target_modules must be a comma-separated subset of ['qkv', 'proj']."
            )

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
        "use_adaln_head": use_adaln_head,
        "adaln_hidden_dim": adaln_hidden_dim,
        "adaln_conditioning_mode": adaln_conditioning_mode,
        "use_dino_lora": use_dino_lora,
        "dino_lora_rank": dino_lora_rank,
        "dino_lora_alpha": dino_lora_alpha,
        "dino_lora_dropout": dino_lora_dropout,
        "dino_lora_num_blocks": dino_lora_num_blocks,
        "dino_lora_target_modules": dino_lora_target_modules,
        "policy_head_type": policy_head_type,
        "diffusion_num_steps": diffusion_num_steps,
        "diffusion_hidden_dim": diffusion_hidden_dim,
        "diffusion_timestep_dim": diffusion_timestep_dim,
        "volumedp_volume_bounds": volumedp_volume_bounds,
        "volumedp_grid_shape": volumedp_grid_shape,
        "volumedp_num_spatial_tokens": volumedp_num_spatial_tokens,
        "volumedp_decoder_layers": volumedp_decoder_layers,
        "volumedp_decoder_heads": volumedp_decoder_heads,
        "volumedp_action_token_dim": volumedp_action_token_dim,
        "proprio_visual_fusion_mode": proprio_visual_fusion_mode,
        "proprio_visual_fusion_hidden_dim": proprio_visual_fusion_hidden_dim,
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
