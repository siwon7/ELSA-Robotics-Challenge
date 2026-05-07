from __future__ import annotations


SUPPORTED_SERVER_STRATEGIES = {
    "fedavg",
    "fedexp",
    "fednova",
    "qfedavg",
    "afl",
    "maxfl",
}


FEDERATED_METHOD_PRESETS = {
    "legacy_auto": {
        "server_strategy": "fedavg",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 50,
        "recommended_rounds": 30,
        "notes": "Backward-compatible mode. Keeps old behavior unless overridden.",
    },
    "fedavg": {
        "server_strategy": "fedavg",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": "Pure FedAvg. Only useful as a clean ablation baseline.",
    },
    "fedprox": {
        "server_strategy": "fedavg",
        "client_update": "proximal_objective",
        "local_regularizer": "fedprox",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": "Default robust choice when client drift is non-trivial.",
    },
    "fedprox_visual_shift": {
        "server_strategy": "fedavg",
        "client_update": "proximal_objective",
        "local_regularizer": "fedprox",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "Recommended preset for strong color/background/camera variation.",
    },
    "fedexp": {
        "server_strategy": "fedexp",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": "FedAvg-compatible client update with adaptive bounded server LR.",
    },
    "fedprox_fedexp": {
        "server_strategy": "fedexp",
        "client_update": "proximal_objective",
        "local_regularizer": "fedprox",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "FedProx local objective with FedExp-style adaptive server LR.",
    },
    "fednova": {
        "server_strategy": "fednova",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": "Normalize client deltas by local steps for computational heterogeneity.",
    },
    "fedprox_fednova": {
        "server_strategy": "fednova",
        "client_update": "proximal_objective",
        "local_regularizer": "fedprox",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "FedProx local objective with FedNova normalized aggregation.",
    },
    "qfedavg": {
        "server_strategy": "qfedavg",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "q-FFL/q-FedAvg-style loss-aware server update for hard clients.",
    },
    "afl": {
        "server_strategy": "afl",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "Agnostic FL-style server dual weights over observed clients/domains.",
    },
    "maxfl": {
        "server_strategy": "maxfl",
        "client_update": "standard",
        "local_regularizer": "none",
        "parameter_scope": "all_trainable",
        "client_state": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "MaxFL-inspired threshold/appeal weighting for client benefit analysis.",
    },
    "fedper_head": {
        "server_strategy": "fedavg",
        "client_update": "local_personal_head",
        "local_regularizer": "none",
        "parameter_scope": "shared_body_local_head",
        "client_state": "local_parameters",
        "local_parameter_keywords": [
            "policy_fc2",
            "diffusion_head",
            "multitoken_diffusion_head",
            "gripper_head",
        ],
        "prox_mu": 0.0,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": (
            "FedPer/FedRep-style split: aggregate shared trainable body, keep "
            "policy/diffusion/gripper heads local per client."
        ),
    },
    "fedprox_fedper_head": {
        "server_strategy": "fedavg",
        "client_update": "proximal_local_personal_head",
        "local_regularizer": "fedprox",
        "parameter_scope": "shared_body_local_head",
        "client_state": "local_parameters",
        "local_parameter_keywords": [
            "policy_fc2",
            "diffusion_head",
            "multitoken_diffusion_head",
            "gripper_head",
        ],
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": (
            "FedProx on shared parameters with a persistent local action head. "
            "Use for controller/head heterogeneity after global-only baselines."
        ),
    },
}


def get_federated_method_preset(config) -> str:
    federated_cfg = getattr(config, "federated", None)
    explicit = getattr(federated_cfg, "method_preset", None)
    if explicit in (None, ""):
        return "legacy_auto"
    preset = str(explicit)
    if preset not in FEDERATED_METHOD_PRESETS:
        raise ValueError(
            f"Unsupported federated.method_preset: {preset}. "
            f"Expected one of {sorted(FEDERATED_METHOD_PRESETS)}"
        )
    return preset


def get_federated_method_spec(config) -> dict:
    spec = dict(FEDERATED_METHOD_PRESETS[get_federated_method_preset(config)])
    federated_cfg = getattr(config, "federated", None)
    server_strategy_override = getattr(federated_cfg, "server_strategy", None)
    if server_strategy_override not in (None, ""):
        spec["server_strategy"] = str(server_strategy_override)
    if spec["server_strategy"] not in SUPPORTED_SERVER_STRATEGIES:
        raise ValueError(
            f"Unsupported federated.server_strategy: {spec['server_strategy']}. "
            f"Expected one of {sorted(SUPPORTED_SERVER_STRATEGIES)}"
        )
    return spec


def get_server_strategy_name(config) -> str:
    return str(get_federated_method_spec(config)["server_strategy"])


def resolve_prox_mu(config, explicit_override=None) -> float:
    if explicit_override not in (None, ""):
        return float(explicit_override)
    federated_cfg = getattr(config, "federated", None)
    federated_value = getattr(federated_cfg, "prox_mu", None)
    if federated_value not in (None, ""):
        return float(federated_value)
    model_value = getattr(getattr(config, "model", None), "prox_mu", None)
    if model_value not in (None, ""):
        return float(model_value)
    return float(get_federated_method_spec(config)["prox_mu"])
