from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import wandb
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server import client_proxy
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAdam, FedAvg, FedAvgM, Krum
from omegaconf import OmegaConf

from elsa_learning_agent.agent_forward_kinematics import Agent


class SaveModelMixin:
    def _parse_save_rounds(self, config: OmegaConf) -> Optional[set[int]]:
        save_rounds = config.dataset.get("save_rounds")
        if save_rounds is None:
            return None
        if isinstance(save_rounds, str):
            rounds = [item.strip() for item in save_rounds.split(",") if item.strip()]
            return {int(item) for item in rounds}
        return {int(item) for item in save_rounds}

    def _init_save_model_mixin(
        self,
        *,
        agent: Agent,
        save_path: Path,
        config: OmegaConf,
        use_wandb: bool,
        strategy_name: str,
        fraction_fit: float,
        resume: bool,
    ) -> None:
        self.agent = agent
        self.strategy_name = strategy_name
        self.save_name = (
            f"{strategy_name}_{agent.policy.__class__.__name__}"
            f"_l-ep_{config.dataset['local_epochs']}"
            f"_ts_{config.dataset['train_split']}_fclients_{fraction_fit}"
        )
        self.save_rounds = self._parse_save_rounds(config)
        self.num_server_rounds = int(config.dataset["num_server_rounds"])
        self.save_path = Path.joinpath(save_path, config.dataset["task"])
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        if not self.use_wandb:
            return

        wandb_kwargs = {
            "project": "BCPolicy-Training",
            "name": f"{config.dataset['task']}_{self.save_name}",
            "config": {
                "strategy": strategy_name,
                "client_epochs": config.dataset["local_epochs"],
                "train_split": config.dataset["train_split"],
                "fitted_clients": fraction_fit,
                "batch_size": config.dataset["batch_size"],
                "rounds": config.dataset["num_server_rounds"],
                "learning_rate": config.model["learning_rate"],
                "weight_decay": config.model["weight_decay"],
            },
        }
        if resume:
            wandb_kwargs["id"] = "run-20250226_101133-uoligt1l"
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[client_proxy.ClientProxy, FitRes]],
        failures: list[Union[tuple[client_proxy.ClientProxy, FitRes], BaseException]],
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )
            params_dict = zip(
                self.agent.federated_state_keys(),
                aggregated_ndarrays,
            )
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.agent.load_federated_state_dict(state_dict)

            should_save = (
                self.save_rounds is None
                or server_round in self.save_rounds
                or server_round == self.num_server_rounds
            )
            if should_save:
                path = Path.joinpath(
                    self.save_path,
                    f"{self.save_name}_round_{server_round}.pth",
                )
                torch.save(self.agent.policy.state_dict(), path)

            if self.use_wandb and aggregated_metrics is not None:
                wandb.log(
                    {
                        "server_round": server_round,
                        "federated_training_loss": aggregated_metrics.get("train_loss"),
                    }
                )

        return aggregated_parameters, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None

        centr_loss, centr_metric = super().evaluate(server_round, parameters)

        if self.use_wandb:
            centralized_eval_metric = {
                "loss_per_env": centr_metric["loss_per_env"],
            }
            if centr_metric.get("avg_reward") is not None:
                centralized_eval_metric["avg_reward"] = centr_metric["avg_reward"]
                centralized_eval_metric["video_array"] = [
                    wandb.Video(video, fps=20, format="gif")
                    for video in centr_metric["video_array"]
                ]

            wandb.log(
                {
                    "server_round": server_round,
                    "centralized_eval_loss": centr_loss,
                    "centralized_eval_metric": centralized_eval_metric,
                }
            )

        return centr_loss, centr_metric

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        fed_eval_loss, fed_eval_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if self.use_wandb:
            wandb.log(
                {
                    "server_round": server_round,
                    "federated_eval_loss": fed_eval_loss,
                }
            )

        return fed_eval_loss, fed_eval_metrics


class SaveFedAvgStrategy(SaveModelMixin, FedAvg):
    def __init__(self, *, agent: Agent, save_path: Path = Path("model_checkpoints"), config: Optional[OmegaConf] = None, use_wandb: bool = False, resume: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._init_save_model_mixin(
            agent=agent,
            save_path=save_path,
            config=config,
            use_wandb=use_wandb,
            strategy_name="fedavg",
            fraction_fit=kwargs["fraction_fit"],
            resume=resume,
        )


class SaveFedAvgMStrategy(SaveModelMixin, FedAvgM):
    def __init__(self, *, agent: Agent, save_path: Path = Path("model_checkpoints"), config: Optional[OmegaConf] = None, use_wandb: bool = False, resume: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._init_save_model_mixin(
            agent=agent,
            save_path=save_path,
            config=config,
            use_wandb=use_wandb,
            strategy_name="fedavgm",
            fraction_fit=kwargs["fraction_fit"],
            resume=resume,
        )


class SaveFedAdamStrategy(SaveModelMixin, FedAdam):
    def __init__(self, *, agent: Agent, save_path: Path = Path("model_checkpoints"), config: Optional[OmegaConf] = None, use_wandb: bool = False, resume: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._init_save_model_mixin(
            agent=agent,
            save_path=save_path,
            config=config,
            use_wandb=use_wandb,
            strategy_name="fedopt",
            fraction_fit=kwargs["fraction_fit"],
            resume=resume,
        )


class SaveKrumStrategy(SaveModelMixin, Krum):
    def __init__(self, *, agent: Agent, save_path: Path = Path("model_checkpoints"), config: Optional[OmegaConf] = None, use_wandb: bool = False, resume: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._init_save_model_mixin(
            agent=agent,
            save_path=save_path,
            config=config,
            use_wandb=use_wandb,
            strategy_name="krum",
            fraction_fit=kwargs["fraction_fit"],
            resume=resume,
        )


def build_strategy(
    strategy_name: str,
    *,
    fraction_fit: float,
    fraction_evaluate: float,
    min_available_clients: int,
    initial_parameters,
    agent: Agent,
    config: OmegaConf,
    use_wandb: bool,
    evaluate_fn=None,
    evaluate_metrics_aggregation_fn=None,
    fit_aggregation_fn=None,
    resume: bool = False,
):
    common_kwargs = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "evaluate_metrics_aggregation_fn": evaluate_metrics_aggregation_fn,
        "fit_metrics_aggregation_fn": fit_aggregation_fn,
        "agent": agent,
        "config": config,
        "use_wandb": use_wandb,
        "resume": resume,
    }

    name = strategy_name.lower()
    if name == "fedavg":
        return SaveFedAvgStrategy(**common_kwargs)
    if name == "fedavgm":
        return SaveFedAvgMStrategy(
            server_learning_rate=1.0,
            server_momentum=0.9,
            **common_kwargs,
        )
    if name == "fedopt":
        return SaveFedAdamStrategy(
            eta=0.1,
            eta_l=config.model["learning_rate"],
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-9,
            **common_kwargs,
        )
    if name == "krum":
        return SaveKrumStrategy(
            num_malicious_clients=0,
            num_clients_to_keep=0,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")
