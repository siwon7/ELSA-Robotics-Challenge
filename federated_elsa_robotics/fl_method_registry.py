from __future__ import annotations


FEDERATED_METHOD_PRESETS = {
    "legacy_auto": {
        "server_strategy": "fedavg",
        "local_regularizer": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 50,
        "recommended_rounds": 30,
        "notes": "Backward-compatible mode. Keeps old behavior unless overridden.",
    },
    "fedavg": {
        "server_strategy": "fedavg",
        "local_regularizer": "none",
        "prox_mu": 0.0,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": "Pure FedAvg. Only useful as a clean ablation baseline.",
    },
    "fedprox": {
        "server_strategy": "fedavg",
        "local_regularizer": "fedprox",
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 10,
        "recommended_rounds": 20,
        "notes": "Default robust choice when client drift is non-trivial.",
    },
    "fedprox_visual_shift": {
        "server_strategy": "fedavg",
        "local_regularizer": "fedprox",
        "prox_mu": 1.0e-3,
        "recommended_local_epochs": 5,
        "recommended_rounds": 20,
        "notes": "Recommended preset for strong color/background/camera variation.",
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
    return FEDERATED_METHOD_PRESETS[get_federated_method_preset(config)]


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
