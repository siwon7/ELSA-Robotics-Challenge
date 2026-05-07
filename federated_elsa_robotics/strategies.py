from typing import Optional, Union
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays, Parameters, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server import client_proxy
from flwr.server.strategy import FedAvg
from pathlib import Path
from omegaconf import OmegaConf
import torch
import wandb
from elsa_learning_agent.agent import Agent
from federated_elsa_robotics.fl_method_registry import get_server_strategy_name
from federated_elsa_robotics.server_aggregation import (
    ClientUpdate,
    ServerSideAggregator,
    client_key_from_metrics,
)
from federated_elsa_robotics.task import get_weights, set_weights


def normalize_run_tag(raw_tag) -> str:
    tag = str(raw_tag or "").strip()
    if not tag:
        return ""
    return tag.replace(" ", "-")


class SaveModelStrategy(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_clients: int = 2,
        initial_parameters = None,
        agent: Optional[Agent] = None,
        save_path: Path = Path("model_checkpoints"),
        config: Optional[OmegaConf] = None,
        use_wandb: bool = False,
        evaluate_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        fit_aggregation_fn: Optional[callable] = None,
        resume: bool = False,
    ):
            
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_aggregation_fn,
        )
        if agent is None:
            raise ValueError("SaveModelStrategy requires an initialized Agent instance.")
        self.agent = agent
        self.config = config
        runtime_cfg = config.get("runtime", {}) if config is not None else {}
        run_tag = normalize_run_tag(runtime_cfg.get("run_tag", ""))
        self.wandb_project = runtime_cfg.get("wandb_project", "BCPolicy-Training")
        name_parts = [
            agent.policy.__class__.__name__,
            f"l-ep_{config.dataset['local_epochs']}",
            f"ts_{config.dataset['train_split']}",
            f"fclients_{fraction_fit}",
        ]
        if run_tag:
            name_parts.append(run_tag)
        self.save_name = "_".join(name_parts)
        # Chkpt/task/wandbName
        self.save_path = Path.joinpath(save_path, config.dataset["task"])
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        config_snapshot_path = Path.joinpath(self.save_path, f"{self.save_name}.config.yaml")
        OmegaConf.save(config, config_snapshot_path)

        if self.use_wandb:
            if resume:
                wandb.init(
                    project=self.wandb_project,
                    # name=f"{base_cfg.dataset.task}_bc_env{idx_environment}_{train_split}",       # Replace with a run name
                    name=f"{config.dataset['task']}_{self.save_name}",
                    id="run-20250226_101133-uoligt1l",
                    resume="must",
                    config={
                        "client_epochs": config.dataset["local_epochs"],
                        "train_split": config.dataset["train_split"],
                        "fitted_clients": fraction_fit,
                        "batch_size": config.dataset["batch_size"],
                        "rounds": config.dataset["num_server_rounds"],
                        "learning_rate": config.model["learning_rate"],
                        "weight_decay": config.model["weight_decay"],
                    }
                )
            else:
                wandb.init(
                    project=self.wandb_project,
                    # name=f"{base_cfg.dataset.task}_bc_env{idx_environment}_{train_split}",       # Replace with a run name
                    name=f"{config.dataset['task']}_{self.save_name}",
                    config={
                        "client_epochs": config.dataset["local_epochs"],
                        "train_split": config.dataset["train_split"],
                        "fitted_clients": fraction_fit,
                        "batch_size": config.dataset["batch_size"],
                        "rounds": config.dataset["num_server_rounds"],
                        "learning_rate": config.model["learning_rate"],
                        "weight_decay": config.model["weight_decay"],
                    }
                )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[client_proxy.ClientProxy, FitRes]],
        failures: list[Union[tuple[client_proxy.ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            set_weights(self.agent, aggregated_ndarrays, config=self.config)

            # Save the model to disk
            path = Path.joinpath(self.save_path, f"{self.save_name}_round_{server_round}.pth")
            torch.save(self.agent.policy.state_dict(), path)

            # Save the parameters to wandb
            if self.use_wandb:
                wandb.log({
                    "server_round": server_round,
                    "federated_training_loss": aggregated_metrics["train_loss"]})

        return aggregated_parameters, aggregated_metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """ Centralized evaluation callback"""

        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        centr_loss, centr_metric = super().evaluate(server_round, parameters)

        # Log the centralized evaluation loss to wandb
        if self.use_wandb:
            centralized_eval_metric = {
                "loss_per_env": centr_metric["loss_per_env"],
            }
            if centr_metric.get("avg_reward") is not None:
                centralized_eval_metric["avg_reward"] = centr_metric["avg_reward"]
                centralized_eval_metric["video_array"] = [
                    wandb.Video(video, fps=20, format="gif") for video in centr_metric["video_array"]
                ]

            wandb.log({
                    "server_round": server_round,
                    "centralized_eval_loss": centr_loss,
                    "centralized_eval_metric": centralized_eval_metric
                    })

        return centr_loss, centr_metric
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation metrics from all clients"""
        fed_eval_loss, fed_eval_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Log the federated evaluation loss to wandb
        if self.use_wandb:
            wandb.log({
                        "server_round": server_round,
                        "federated_eval_loss": fed_eval_loss
                    })

        return fed_eval_loss, fed_eval_metrics


class TrainableOnlyFederatedStrategy(SaveModelStrategy):
    """Checkpointing strategy for trainable-only FL aggregation surfaces."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_strategy_name = get_server_strategy_name(self.config)
        self.server_aggregator = ServerSideAggregator(
            self.server_strategy_name,
            config=self.config,
        )

    def _aggregate_fit_metrics(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        strategy_metrics: dict[str, Scalar],
    ) -> dict[str, Scalar]:
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            print("WARNING: No fit_metrics_aggregation_fn provided")
        metrics_aggregated.update(strategy_metrics)
        return metrics_aggregated

    def _save_trainable_checkpoint(
        self,
        server_round: int,
        aggregated_parameters: Parameters,
        aggregated_metrics: dict[str, Scalar],
    ) -> None:
        print(
            f"Saving round {server_round} aggregated_parameters "
            f"strategy={self.server_strategy_name}..."
        )
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        set_weights(self.agent, aggregated_ndarrays, config=self.config)

        path = Path.joinpath(self.save_path, f"{self.save_name}_round_{server_round}.pth")
        torch.save(self.agent.policy.state_dict(), path)

        if self.use_wandb:
            payload = {
                "server_round": server_round,
                "server_strategy": self.server_strategy_name,
            }
            if "train_loss" in aggregated_metrics:
                payload["federated_training_loss"] = aggregated_metrics["train_loss"]
            for key in (
                "server_lr",
                "fednova_effective_tau",
                "qffl_q",
                "afl_lambda_max",
                "maxfl_weight_max",
            ):
                if key in aggregated_metrics:
                    payload[key] = aggregated_metrics[key]
            wandb.log(payload)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        previous_arrays = get_weights(self.agent, config=self.config)
        updates = [
            ClientUpdate(
                arrays=parameters_to_ndarrays(fit_res.parameters),
                num_examples=int(fit_res.num_examples),
                metrics=dict(fit_res.metrics),
                client_key=client_key_from_metrics(
                    dict(fit_res.metrics),
                    fallback=f"client_{index}",
                ),
            )
            for index, (_, fit_res) in enumerate(results)
        ]
        aggregated_ndarrays, strategy_metrics = self.server_aggregator.aggregate(
            updates,
            previous_arrays,
        )
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        aggregated_metrics = self._aggregate_fit_metrics(
            server_round,
            results,
            strategy_metrics,
        )
        self._save_trainable_checkpoint(
            server_round,
            aggregated_parameters,
            aggregated_metrics,
        )
        return aggregated_parameters, aggregated_metrics
