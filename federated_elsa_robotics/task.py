"""elsa-robotics: A Flower / PyTorch app."""

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.agent import Agent


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


def iter_trainable_parameters(agent: Agent) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable policy parameters in a deterministic order."""
    return [
        (name, param)
        for name, param in agent.policy.named_parameters()
        if param.requires_grad
    ]


# Training and validation loop
def train_one_epoch(
    agent: Agent,
    train_loader,
    optimizer,
    criterion,
    epoch,
    device,
    prox_mu: float = 0.0,
    global_trainable_params: Iterable[torch.Tensor] | None = None,
):
    agent.train()
    total_loss = 0.0
    for batch in train_loader:
        image = batch["image"].to(device)
        low_dim_state = batch["low_dim_state"].to(device)
        action = batch["action"].to(device)

        optimizer.zero_grad()
        loss = agent.compute_loss(image, low_dim_state, action, criterion=criterion)

        total_objective = loss
        if prox_mu > 0.0 and global_trainable_params is not None:
            proximal_term = torch.zeros((), device=device)
            for (_, local_param), global_param in zip(
                iter_trainable_parameters(agent),
                global_trainable_params,
            ):
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

            predicted_action = agent.get_action(image, low_dim_state)
            loss = criterion(predicted_action, action)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    # print(f"Epoch {epoch}: Validation Loss = {avg_loss:.4f}")
    return avg_loss

def train(agent: Agent, trainloader, epochs, device, config):
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
            param.detach().clone()
            for param in trainable_params
        ]

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
    return avg_trainloss

def get_weights(agent: Agent):
    return [
        param.detach().cpu().numpy()
        for _, param in iter_trainable_parameters(agent)
    ]

def set_weights(agent: Agent, parameters: list):
    trainable_params = iter_trainable_parameters(agent)
    if len(parameters) != len(trainable_params):
        raise ValueError(
            f"Expected {len(trainable_params)} trainable tensors, got {len(parameters)}"
        )

    with torch.no_grad():
        for (name, param), incoming in zip(trainable_params, parameters):
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
