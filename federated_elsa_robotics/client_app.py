"""elsa-robotics: A Flower / PyTorch app."""

from pathlib import Path

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
    resolve_run_config,
    set_global_seed,
)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self,
        net,
        trainloader,
        valloader,
        local_epochs,
        config=None,
        local_state_path: Path | None = None,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.policy.to(self.device)
        self.config = config
        self.local_state_path = local_state_path
        self._restore_local_state()

    def _restore_local_state(self):
        if self.local_state_path is None or not self.local_state_path.exists():
            return
        local_state = torch.load(self.local_state_path, map_location="cpu")
        try:
            self.net.load_local_state_dict(local_state)
        except RuntimeError:
            print(f"Skipping incompatible local state: {self.local_state_path}")

    def _save_local_state(self):
        if self.local_state_path is None:
            return
        local_state = self.net.get_local_state_dict()
        if not local_state:
            return
        self.local_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(local_state, self.local_state_path)

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
        self._save_local_state()

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
    run_config = resolve_run_config(context.run_config)

    # Load the dataset configuration from the yaml file
    config = OmegaConf.load(run_config["dataset-config-path"])
    config.dataset.env_id = partition_id
    config.dataset.dataset_task = run_config["dataset-task"]
    config.dataset.task = run_config["dataset-task"]
    config.dataset.train_split = run_config["train-split"]
    policy_name = get_runtime_policy_name(run_config, config)
    config.model.policy_name = policy_name
    seed = int(run_config.get("seed", DEFAULT_SEED))
    deterministic = parse_bool(run_config.get("deterministic-training", False))
    set_global_seed(seed + int(partition_id), deterministic)
    local_state_root = Path("results") / "client_local_state"
    enable_client_local_state = parse_bool(
        run_config.get("enable-client-local-state", True)
    )
    
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
    local_epochs = run_config["local-epochs"]
    local_state_path = None
    if enable_client_local_state:
        local_state_path = (
            local_state_root
            / policy_name
            / config.dataset.task
            / f"client_{partition_id:03d}.pth"
        )

    # Return Client instance
    return FlowerClient(
        agent,
        trainloader,
        valloader,
        local_epochs,
        config=config,
        local_state_path=local_state_path,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
