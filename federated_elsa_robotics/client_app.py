"""elsa-robotics: A Flower / PyTorch app."""

from omegaconf import OmegaConf
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_elsa_robotics.task import get_weights, load_data_colosseum, set_weights, train, validate_one_epoch
from federated_elsa_robotics.policy_runtime import (
    DEFAULT_SEED,
    build_runtime_agent,
    get_runtime_policy_name,
    parse_bool,
    set_global_seed,
)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, config=None):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load the dataset configuration from the yaml file
    config = OmegaConf.load(context.run_config["dataset-config-path"])
    config.dataset.env_id = partition_id
    config.dataset.dataset_task = context.run_config["dataset-task"]
    config.dataset.task = context.run_config["dataset-task"]
    config.dataset.train_split = context.run_config["train-split"]
    policy_name = get_runtime_policy_name(context.run_config, config)
    config.model.policy_name = policy_name
    seed = int(context.run_config.get("seed", DEFAULT_SEED))
    deterministic = parse_bool(context.run_config.get("deterministic-training", False))
    set_global_seed(seed + int(partition_id), deterministic)
    
    # Load the data
    trainloader, valloader = load_data_colosseum(partition_id, num_partitions, config=config)
    sample = next(iter(trainloader))
    agent = build_runtime_agent(
        policy_name=policy_name,
        image_size=(sample["image"].shape[2], sample["image"].shape[3]),
        action_dim=sample["action"].shape[1],
        low_dim_state_dim=sample["low_dim_state"].shape[1],
        config=config,
    )
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(agent, trainloader, valloader, local_epochs, config=config).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
