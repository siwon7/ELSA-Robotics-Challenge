"""elsa-robotics: A Flower / PyTorch app."""

from flwr.common import Context, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from pathlib import Path
from federated_elsa_robotics.strategies import SaveModelStrategy
from federated_elsa_robotics.task import (
    get_weights,
    infer_action_dim,
    set_weights,
)
from omegaconf import OmegaConf
import torch
from typing import Optional, Union

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.config_validation import validate_runtime_config
from federated_elsa_robotics.fl_method_registry import resolve_prox_mu


class TrainableOnlySaveModelStrategy(SaveModelStrategy):
    """Save strategy compatible with trainable-only aggregation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, object]],
        failures: list[Union[tuple[ClientProxy, object], BaseException]],
    ) -> tuple[Optional[object], dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = FedAvg.aggregate_fit(
            self,
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            set_weights(self.agent, aggregated_ndarrays)

            path = Path.joinpath(self.save_path, f"{self.save_name}_round_{server_round}.pth")
            torch.save(self.agent.policy.state_dict(), path)

            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "server_round": server_round,
                        "federated_training_loss": aggregated_metrics["train_loss"],
                    }
                )

        return aggregated_parameters, aggregated_metrics
def train_aggregation_fn(metrics: list[dict]):
    """Aggregate training metrics."""
    losses = [num_samples * m["train_loss"] for num_samples, m in metrics]
    examples = [num_samples for num_samples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}

def server_fn(context: Context):
    # Read from project toml config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-eval"]
    server_device = context.run_config["server-device"]
    client_device = context.run_config["client-device"]
    use_wandb = context.run_config["use-wandb"]
    wandb_project = context.run_config["wandb-project"]
    checkpoint_root = context.run_config["checkpoint-root"]
    run_tag = context.run_config["run-tag"]
    dataset_config_path = context.run_config["dataset-config-path"]
    prox_mu_override = context.run_config.get("prox-mu", "")
    conf = OmegaConf.load(dataset_config_path)
    conf.model.prox_mu = resolve_prox_mu(conf, explicit_override=prox_mu_override)
    validation_summary = validate_runtime_config(conf)
    print(
        f"Starting server with l-ep={context.run_config['local-epochs']}, "
        f"ts={context.run_config['train-split']}, fclients={fraction_fit}, "
        f"prox_mu={conf.model.prox_mu}, fl={validation_summary['federated_method_preset']}"
    )
    conf.dataset.action_dim = infer_action_dim(conf)

    net_args = {
        "image_channels": 3,
        "low_dim_state_dim": 8,
        "action_dim": int(conf.dataset.action_dim),
        "image_size": (128, 128),
        **get_agent_model_kwargs(conf),
    }

    # Initialize model parameters
    agent = Agent(**net_args)
    ndarrays = get_weights(agent)
    parameters = ndarrays_to_parameters(ndarrays)

    # Evaluation loader
    def create_config(idx): 
        cur_config = conf.copy()
        cur_config.dataset.task = context.run_config["dataset-task"]
        cur_config.dataset.env_id = idx
        # Use evaluation dataset for the server
        cur_config.dataset.root_dir = cur_config.dataset.root_eval_dir
        cur_config.dataset.test_split = 0.0
        cur_config.dataset.train_split = context.run_config["train-split"]
        cur_config.dataset.num_server_rounds = num_rounds
        cur_config.dataset.local_epochs = context.run_config["local-epochs"]
        cur_config.dataset.action_dim = infer_action_dim(cur_config)
        cur_config.model.prox_mu = float(conf.model.prox_mu)
        return cur_config
    config = create_config(0)
    config.runtime = {
        "server_device": server_device,
        "client_device": client_device,
        "wandb_project": wandb_project,
        "checkpoint_root": checkpoint_root,
        "run_tag": run_tag,
        "prox_mu": float(conf.model.prox_mu),
    }

    strategy = TrainableOnlySaveModelStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=parameters,
        agent=agent,
        save_path=Path(checkpoint_root),
        config=config,
        use_wandb=use_wandb,
        fit_aggregation_fn=train_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def get_server_app():
    return ServerApp(server_fn=server_fn)
# Create ServerApp
app = ServerApp(server_fn=server_fn)
