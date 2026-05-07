from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import hashlib
import json

import torch

from federated_elsa_robotics.fl_method_registry import get_federated_method_spec


def _as_string_list(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(token).strip() for token in value if str(token).strip()]


def get_local_parameter_keywords(config) -> list[str]:
    """Return parameter-name tokens that should remain client-local."""
    federated_cfg = getattr(config, "federated", None)
    override = getattr(federated_cfg, "local_parameter_keywords", None)
    if override not in (None, ""):
        return _as_string_list(override)
    spec = get_federated_method_spec(config)
    return _as_string_list(spec.get("local_parameter_keywords", []))


def uses_local_parameter_state(config) -> bool:
    """Whether selected trainable parameters should persist only on clients."""
    spec = get_federated_method_spec(config)
    return str(spec.get("client_state", "none")) == "local_parameters"


def is_local_only_parameter(name: str, config) -> bool:
    if not uses_local_parameter_state(config):
        return False
    return any(keyword in name for keyword in get_local_parameter_keywords(config))


def iter_trainable_policy_parameters(agent) -> list[tuple[str, torch.nn.Parameter]]:
    """Return all trainable policy parameters in a deterministic order."""
    return [
        (name, param)
        for name, param in agent.policy.named_parameters()
        if param.requires_grad
    ]


def iter_aggregated_parameters(agent, config=None) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable parameters sent to/aggregated by the server."""
    params = iter_trainable_policy_parameters(agent)
    if config is None:
        return params
    return [
        (name, param)
        for name, param in params
        if not is_local_only_parameter(name, config)
    ]


def iter_local_only_parameters(agent, config=None) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable parameters that remain persistent client-local state."""
    if config is None:
        return []
    return [
        (name, param)
        for name, param in iter_trainable_policy_parameters(agent)
        if is_local_only_parameter(name, config)
    ]


def _tensor_manifest(params: Iterable[tuple[str, torch.nn.Parameter]]) -> list[dict]:
    return [
        {
            "name": name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "dtype": str(param.dtype),
        }
        for name, param in params
    ]


def get_parameter_surface_manifest(agent, config=None) -> dict:
    """Summarize all trainable, aggregated, and local-only parameter surfaces."""
    all_policy_params = list(agent.policy.named_parameters())
    trainable_params = iter_trainable_policy_parameters(agent)
    aggregated_params = iter_aggregated_parameters(agent, config)
    local_only_params = iter_local_only_parameters(agent, config)
    total_params = sum(param.numel() for _, param in all_policy_params)
    trainable_count = sum(param.numel() for _, param in trainable_params)
    aggregated_count = sum(param.numel() for _, param in aggregated_params)
    local_only_count = sum(param.numel() for _, param in local_only_params)
    tensors = _tensor_manifest(aggregated_params)
    local_only_tensors = _tensor_manifest(local_only_params)
    return {
        "num_trainable_tensors": len(tensors),
        "trainable_params": int(aggregated_count),
        "total_policy_params": int(total_params),
        "total_trainable_policy_params": int(trainable_count),
        "local_only_params": int(local_only_count),
        "trainable_fraction": (
            float(aggregated_count) / float(total_params)
            if total_params > 0
            else 0.0
        ),
        "local_only_fraction": (
            float(local_only_count) / float(total_params)
            if total_params > 0
            else 0.0
        ),
        "parameter_scope": get_federated_method_spec(config)["parameter_scope"]
        if config is not None
        else "all_trainable",
        "local_parameter_keywords": get_local_parameter_keywords(config)
        if config is not None
        else [],
        # Backward-compatible key: these are the tensors the server aggregates.
        "tensors": tensors,
        "aggregated_tensors": tensors,
        "local_only_tensors": local_only_tensors,
    }


def get_manifest_hash(manifest: dict) -> str:
    """Return a stable hash for the aggregated parameter surface."""
    payload = json.dumps(manifest.get("aggregated_tensors", []), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def get_parameter_arrays(agent, config=None) -> list:
    return [
        param.detach().cpu().numpy()
        for _, param in iter_aggregated_parameters(agent, config)
    ]


def set_parameter_arrays(agent, parameters: list, config=None) -> None:
    aggregated_params = iter_aggregated_parameters(agent, config)
    if len(parameters) != len(aggregated_params):
        raise ValueError(
            f"Expected {len(aggregated_params)} aggregated tensors, got {len(parameters)}"
        )

    with torch.no_grad():
        for (name, param), incoming in zip(aggregated_params, parameters):
            incoming_tensor = torch.as_tensor(
                incoming,
                dtype=param.dtype,
                device=param.device,
            )
            if tuple(incoming_tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(incoming_tensor.shape)}"
                )
            param.copy_(incoming_tensor)


def save_local_only_state(agent, config, path: Path) -> int:
    """Persist client-local parameters for FedPer/FedRep-style methods."""
    local_params = iter_local_only_parameters(agent, config)
    if not local_params:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: param.detach().cpu()
        for name, param in local_params
    }
    torch.save(state, path)
    return len(state)


def load_local_only_state(agent, config, path: Path, device) -> int:
    """Load previously persisted client-local parameters if present."""
    local_params = iter_local_only_parameters(agent, config)
    if not local_params or not path.exists():
        return 0
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid local parameter state at {path}")
    loaded = 0
    with torch.no_grad():
        for name, param in local_params:
            if name not in state:
                continue
            incoming = torch.as_tensor(state[name], dtype=param.dtype, device=param.device)
            if tuple(incoming.shape) != tuple(param.shape):
                raise ValueError(
                    f"Local state shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(incoming.shape)}"
                )
            param.copy_(incoming)
            loaded += 1
    return loaded
