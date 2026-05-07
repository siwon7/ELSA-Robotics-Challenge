"""elsa-robotics: A Flower / PyTorch app."""

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.agent import Agent
from elsa_learning_agent.utils import move_nested_to_device
from federated_elsa_robotics.parameter_surfaces import (
    get_manifest_hash,
    get_parameter_arrays,
    get_parameter_surface_manifest,
    iter_aggregated_parameters,
    iter_trainable_policy_parameters,
    set_parameter_arrays,
)


def load_data_colosseum(partition_id: int, num_partitions: int, train_split : float = 0.9, config: dict = None):
    """Load partition Colosseum data."""
    # Load the dataset
    train_dataset = ImitationDataset(config=config, train=True, normalize=True)
    print(f"For partition_id={partition_id}, len(train_dataset): {len(train_dataset)}")

    val_dataset = ImitationDataset(config=config, test=True, normalize=True)
    print(f"For partition_id={partition_id}, len(val_dataset): {len(val_dataset)}")

    num_workers = int(getattr(config.dataset, "num_workers", 0) or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def infer_action_dim(config) -> int:
    """Infer action dimensionality from config, including optional chunking."""
    dataset_cfg = getattr(config, "dataset", None)
    transform_cfg = getattr(config, "transform", None)

    explicit_action_dim = getattr(dataset_cfg, "action_dim", None)
    if explicit_action_dim not in (None, ""):
        return int(explicit_action_dim)

    chunk_len = int(getattr(dataset_cfg, "action_chunk_len", 1) or 1)
    base_action_dim = getattr(dataset_cfg, "base_action_dim", None)
    if base_action_dim not in (None, ""):
        return int(base_action_dim) * chunk_len

    action_min = getattr(transform_cfg, "action_min", None)
    bounds_dim = len(action_min) if action_min is not None else 0
    if bounds_dim > 0:
        # Backward-compatible heuristic: existing configs store per-step bounds.
        if chunk_len > 1 and bounds_dim == 8:
            return bounds_dim * chunk_len
        return bounds_dim

    return 8 * chunk_len


def infer_low_dim_state_dim(config) -> int:
    """Infer proprio/state dimensionality from config without loading data."""
    dataset_cfg = getattr(config, "dataset", None)
    base_dim = int(getattr(dataset_cfg, "low_dim_state_dim", 8) or 8)
    if bool(getattr(dataset_cfg, "include_camera_in_state", False)):
        # process_obs appends flattened front K (3x3) and T (4x4).
        base_dim += 9 + 16
    return base_dim


def iter_trainable_parameters(agent: Agent) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable policy parameters in a deterministic order."""
    return iter_trainable_policy_parameters(agent)


def get_trainable_parameter_manifest(agent: Agent, config=None) -> dict:
    """Summarize the exact parameter surface used by FL aggregation."""
    return get_parameter_surface_manifest(agent, config)


def get_trainable_manifest_hash(manifest: dict) -> str:
    """Return a stable hash for the aggregated parameter surface."""
    return get_manifest_hash(manifest)


def estimate_train_loss(
    agent: Agent,
    data_loader,
    device,
    *,
    max_batches: int | None = None,
) -> float:
    """Estimate training objective RMSE without optimizer updates."""
    criterion = nn.MSELoss()
    agent.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in data_loader:
            image = batch["image"].to(device)
            low_dim_state = batch["low_dim_state"].to(device)
            action = batch["action"].to(device)
            obs_context = move_nested_to_device(batch.get("obs_context"), device)
            ee_position = batch.get("ee_position")
            if ee_position is not None:
                ee_position = ee_position.to(device)
            gripper_target_weight = batch.get("gripper_target_weight")
            if gripper_target_weight is not None:
                gripper_target_weight = gripper_target_weight.to(device)

            loss = agent.compute_loss(
                image,
                low_dim_state,
                action,
                criterion=criterion,
                obs_context=obs_context,
                ee_position=ee_position,
                gripper_target_weight=gripper_target_weight,
            )
            total_loss += math.sqrt(loss.item())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    if batches == 0:
        raise ValueError("Cannot estimate loss on an empty data loader")
    return total_loss / batches


# Training and validation loop
def train_one_epoch(
    agent: Agent,
    train_loader,
    optimizer,
    criterion,
    epoch,
    device,
    prox_mu: float = 0.0,
    global_trainable_params: Iterable[tuple[str, torch.Tensor]] | None = None,
):
    agent.train()
    total_loss = 0.0
    for batch in train_loader:
        image = batch["image"].to(device)
        low_dim_state = batch["low_dim_state"].to(device)
        action = batch["action"].to(device)
        obs_context = move_nested_to_device(batch.get("obs_context"), device)
        ee_position = batch.get("ee_position")
        if ee_position is not None:
            ee_position = ee_position.to(device)
        gripper_target_weight = batch.get("gripper_target_weight")
        if gripper_target_weight is not None:
            gripper_target_weight = gripper_target_weight.to(device)

        optimizer.zero_grad()
        loss = agent.compute_loss(
            image,
            low_dim_state,
            action,
            criterion=criterion,
            obs_context=obs_context,
            ee_position=ee_position,
            gripper_target_weight=gripper_target_weight,
        )

        total_objective = loss
        if prox_mu > 0.0 and global_trainable_params is not None:
            proximal_term = torch.zeros((), device=device)
            local_params = dict(iter_trainable_parameters(agent))
            for name, global_param in global_trainable_params:
                local_param = local_params[name]
                proximal_term = proximal_term + torch.sum((local_param - global_param) ** 2)
            total_objective = total_objective + 0.5 * prox_mu * proximal_term

        total_objective.backward()
        optimizer.step()

        total_loss += math.sqrt(loss.item())  # RMSE loss

    avg_loss = total_loss / len(train_loader)
    # print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(agent:Agent, val_loader, device):
    criterion = torch.nn.MSELoss()
    agent.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            low_dim_state = batch["low_dim_state"].to(device)
            action = batch["action"].to(device)
            obs_context = move_nested_to_device(batch.get("obs_context"), device)

            predicted_action = agent.get_action(image, low_dim_state, obs_context=obs_context)
            loss = criterion(predicted_action, action)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    # print(f"Epoch {epoch}: Validation Loss = {avg_loss:.4f}")
    return avg_loss

def train(
    agent: Agent,
    trainloader,
    epochs,
    device,
    config,
    *,
    metrics_probe_batches: int = 0,
    return_metrics: bool = False,
):
    """Train the model on the training set."""
    agent.policy.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()  # Behavioral Cloning uses MSE loss
    trainable_params = [param for _, param in iter_trainable_parameters(agent)]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimization")
    optimizer = optim.Adam(
        trainable_params,
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    prox_mu = float(getattr(config.model, "prox_mu", 0.0) or 0.0)
    global_trainable_params = None
    if prox_mu > 0.0:
        global_trainable_params = [
            (name, param.detach().clone())
            for name, param in iter_aggregated_parameters(agent, config)
        ]

    metrics: dict[str, float | int] = {}
    if return_metrics and metrics_probe_batches > 0:
        metrics["pre_train_loss"] = estimate_train_loss(
            agent,
            trainloader,
            device,
            max_batches=metrics_probe_batches,
        )

    # Training loop
    running_loss = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            agent,
            trainloader,
            optimizer,
            criterion,
            epoch,
            device,
            prox_mu=prox_mu,
            global_trainable_params=global_trainable_params,
        )
        running_loss += train_loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    avg_trainloss = running_loss / epochs
    if return_metrics:
        if metrics_probe_batches > 0:
            metrics["post_train_loss"] = estimate_train_loss(
                agent,
                trainloader,
                device,
                max_batches=metrics_probe_batches,
            )
        metrics["local_steps"] = int(epochs * len(trainloader))
        metrics["num_batches"] = int(len(trainloader))
        return avg_trainloss, metrics
    return avg_trainloss

def get_weights(agent: Agent, config=None):
    return get_parameter_arrays(agent, config)

def get_aggregated_weights(agent: Agent, config=None):
    return get_parameter_arrays(agent, config)


def set_weights(agent: Agent, parameters: list, config=None):
    set_parameter_arrays(agent, parameters, config)


def set_aggregated_weights(agent: Agent, parameters: list, config=None):
    set_parameter_arrays(agent, parameters, config)


def iter_aggregated_trainable_parameters(
    agent: Agent,
    config=None,
) -> list[tuple[str, torch.nn.Parameter]]:
    return iter_aggregated_parameters(agent, config)
