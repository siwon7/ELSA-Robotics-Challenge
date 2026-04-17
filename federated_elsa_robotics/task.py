"""elsa-robotics: A Flower / PyTorch app."""

from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from federated_elsa_robotics.policy_runtime import trim_low_dim_state
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.agent_forward_kinematics import Agent


def auxiliary_loss_weights(config):
    model_cfg = config.model
    return {
        "phase_loss": float(model_cfg.get("phase_loss_weight", 0.0)),
        "relation_loss": float(model_cfg.get("relation_loss_weight", 0.0)),
        "retrieval_loss": float(model_cfg.get("retrieval_loss_weight", 0.0)),
        "recovery_loss": float(model_cfg.get("recovery_loss_weight", 0.0)),
        "smooth_loss": float(model_cfg.get("smooth_loss_weight", 0.0)),
    }


def compute_total_loss(agent, batch, criterion, config):
    image = batch["image"]
    low_dim_state = trim_low_dim_state(agent, batch["low_dim_state"])
    action = batch["action"]
    progress = batch.get("progress")

    predicted_action, aux = agent.get_action(
        image,
        low_dim_state,
        return_aux=True,
        progress=progress,
        action_target=action,
    )
    bc_loss = criterion(predicted_action, action)
    total_loss = bc_loss
    metrics = {"bc_loss": float(bc_loss.detach().item())}

    for name, weight in auxiliary_loss_weights(config).items():
        aux_value = aux.get(name)
        if aux_value is None or weight <= 0.0:
            continue
        total_loss = total_loss + (weight * aux_value)
        metrics[name] = float(aux_value.detach().item())

    return total_loss, metrics


def load_data_colosseum(partition_id: int, num_partitions: int, train_split : float = 0.9, config: dict = None):
    """Load partition Colosseum data."""
    # Load the dataset
    train_dataset = ImitationDataset(config=config, train=True)
    print(f"For partition_id={partition_id}, len(train_dataset): {len(train_dataset)}")

    val_dataset = ImitationDataset(config=config, test=True)
    print(f"For partition_id={partition_id}, len(val_dataset): {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False)
    return train_loader, val_loader

# Training and validation loop
def train_one_epoch(agent: Agent, train_loader, optimizer, criterion, epoch, device):
    agent.train()
    total_loss = 0.0
    for batch in train_loader:
        image = batch["image"].to(device)
        action = batch["action"].to(device)
        train_batch = {
            "image": image,
            "low_dim_state": batch["low_dim_state"].to(device),
            "action": action,
        }
        if "progress" in batch:
            train_batch["progress"] = batch["progress"].to(device)

        optimizer.zero_grad()
        loss, _ = compute_total_loss(
            agent,
            train_batch,
            criterion=criterion,
            config=agent.training_config,
        )
        loss.backward()
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
            low_dim_state = trim_low_dim_state(agent, batch["low_dim_state"].to(device))
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
    agent.training_config = config
    criterion = nn.MSELoss()  # Behavioral Cloning uses MSE loss
    optimizer = optim.Adam(agent.policy.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    # Training loop
    running_loss = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(agent, trainloader, optimizer, criterion, epoch, device)
        running_loss += train_loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    avg_trainloss = running_loss / epochs
    return avg_trainloss

def get_weights(agent: Agent):
    return [val.cpu().numpy() for val in agent.get_federated_state_dict().values()]

def set_weights(agent: Agent, parameters: list):
    params_dict = zip(agent.federated_state_keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    agent.load_federated_state_dict(state_dict)
