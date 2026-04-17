"""elsa-robotics: A Flower / PyTorch app."""

import os

from omegaconf import OmegaConf
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_elsa_robotics.task import (
    get_weights,
    infer_action_dim,
    load_data_colosseum,
    set_weights,
    train,
    validate_one_epoch,
)
from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_validation import validate_runtime_config
from elsa_learning_agent.dataset.path_utils import (
    available_env_ids,
    resolve_dataset_root,
)
from elsa_learning_agent.utils import get_action_output_activation
from federated_elsa_robotics.fl_method_registry import resolve_prox_mu

def resolve_torch_device(device_name: str) -> torch.device:
    """Resolve a run-config device string without forcing clients onto cuda:0."""
    if torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def _run_config_or_env(context: Context, key: str, env_key: str) -> str:
    """Read a Flower run_config value, falling back to environment variables.

    In `flwr run .`, clients receive the full run_config. In programmatic
    simulations (`start_simulation`), client actors may only receive node_config,
    so we also read the values from environment variables exported by the runner.
    """
    if key in context.run_config:
        return context.run_config[key]
    value = os.environ.get(env_key)
    if value is None:
        raise KeyError(key)
    return value


def _run_config_or_env_default(
    context: Context,
    key: str,
    env_key: str,
    default: str,
) -> str:
    try:
        return _run_config_or_env(context, key, env_key)
    except KeyError:
        return default


def resolve_partition_env_id(config, partition_id: int) -> int:
    """Map a partition id to an existing env shard to avoid missing-file crashes."""
    env_ids = available_env_ids(str(config.dataset.root_dir), str(config.dataset.task))
    if partition_id in env_ids:
        return partition_id
    fallback_env_id = env_ids[partition_id % len(env_ids)]
    print(
        f"Partition {partition_id} has no shard for task={config.dataset.task}; "
        f"remapping to env_id={fallback_env_id}"
    )
    return fallback_env_id


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net: Agent, trainloader, valloader, local_epochs, device, config=None):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = device
        self.net.policy.to(self.device)
        self.config = config

    def fit(self, parameters, config):
        print("Training Client...")
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            config=self.config,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        print("Evaluating client...")
        set_weights(self.net, parameters)
        val_loss = validate_one_epoch(self.net, self.valloader, self.device)
        return val_loss, len(self.valloader.dataset), {}
        


def client_fn(context: Context):
    # Load model and data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Load the dataset configuration from the yaml file
    dataset_config_path = _run_config_or_env(context, "dataset-config-path", "ELSA_DATASET_CONFIG_PATH")
    dataset_task = _run_config_or_env(context, "dataset-task", "ELSA_DATASET_TASK")
    train_split = float(_run_config_or_env(context, "train-split", "ELSA_TRAIN_SPLIT"))
    local_epochs = int(_run_config_or_env(context, "local-epochs", "ELSA_LOCAL_EPOCHS"))
    client_device_name = _run_config_or_env(context, "client-device", "ELSA_CLIENT_DEVICE")
    prox_mu_override = _run_config_or_env_default(context, "prox-mu", "ELSA_PROX_MU", "")
    client_device = resolve_torch_device(client_device_name)
    probe_device = str(client_device)
    if client_device.type == "cuda":
        probe = torch.zeros(1, device=client_device)
        probe_device = str(probe.device)

    print(
        "Client bootstrap "
        f"pid={os.getpid()} partition={partition_id} device_req={client_device_name} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"probe_device={probe_device} prox_mu_override={prox_mu_override or '<none>'}"
    )

    config = OmegaConf.load(dataset_config_path)
    config.dataset.dataset_task = dataset_task
    config.dataset.task = dataset_task
    config.dataset.train_split = train_split
    config.dataset.root_dir = resolve_dataset_root(str(config.dataset.root_dir), dataset_task)
    config.dataset.root_eval_dir = resolve_dataset_root(str(config.dataset.root_eval_dir), dataset_task)
    config.dataset.root_test_dir = resolve_dataset_root(str(config.dataset.root_test_dir), dataset_task)
    config.dataset.env_id = resolve_partition_env_id(config, partition_id)
    config.model.prox_mu = resolve_prox_mu(config, explicit_override=prox_mu_override)
    validation_summary = validate_runtime_config(config)
    print(f"Runtime config summary: {validation_summary}")
    config.dataset.action_dim = infer_action_dim(config)
    
    # Load the data
    trainloader, valloader = load_data_colosseum(partition_id, num_partitions, config=config)
    sample = next(iter(trainloader))
    sample_action_dim = int(sample["action"].shape[1])
    action_dim = int(config.dataset.action_dim)
    if action_dim != sample_action_dim:
        print(
            f"Config-derived action_dim={action_dim} does not match sample action_dim="
            f"{sample_action_dim}; using sample dimension"
        )
        action_dim = sample_action_dim
    agent = Agent(
            image_channels=3,
            low_dim_state_dim=sample["low_dim_state"].shape[1],
            action_dim=action_dim,
            image_size=(sample["image"].shape[2], sample["image"].shape[3]),
            vision_backbone=str(getattr(config.model, "vision_backbone", "cnn")),
            projector_dim=int(getattr(config.model, "projector_dim", 256)),
            action_output_activation=get_action_output_activation(config),
            normalize_branch_embeddings=bool(
                getattr(config.model, "normalize_branch_embeddings", False)
            ),
            low_dim_dropout_prob=float(
                getattr(config.model, "low_dim_dropout_prob", 0.0) or 0.0
            ),
        )
    # Return Client instance
    return FlowerClient(
        agent,
        trainloader,
        valloader,
        local_epochs,
        client_device,
        config=config,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
