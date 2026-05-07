"""elsa-robotics: A Flower / PyTorch app."""

import json
from flwr.common import Context, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from pathlib import Path
from federated_elsa_robotics.strategies import TrainableOnlyFederatedStrategy
from federated_elsa_robotics.task import (
    get_weights,
    get_trainable_parameter_manifest,
    infer_action_dim,
    infer_low_dim_state_dim,
)
from omegaconf import OmegaConf

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import get_agent_model_kwargs
from elsa_learning_agent.config_validation import validate_runtime_config
from elsa_learning_agent.utils import get_expected_image_channels
from federated_elsa_robotics.fl_method_registry import resolve_prox_mu


def train_aggregation_fn(metrics: list[tuple[int, dict[str, Scalar]]]):
    """Aggregate training metrics."""
    if not metrics:
        return {}

    def weighted_mean(key: str) -> float | None:
        weighted_values = [
            (num_samples, float(client_metrics[key]))
            for num_samples, client_metrics in metrics
            if key in client_metrics
        ]
        if not weighted_values:
            return None
        total_examples = sum(num_samples for num_samples, _ in weighted_values)
        if total_examples <= 0:
            return None
        return (
            sum(num_samples * value for num_samples, value in weighted_values)
            / total_examples
        )

    aggregated: dict[str, Scalar] = {
        "fit_clients": len(metrics),
    }
    train_loss = weighted_mean("train_loss")
    if train_loss is not None:
        aggregated["train_loss"] = train_loss

    for key in (
        "pre_train_loss",
        "post_train_loss",
        "delta_norm_sq",
        "local_steps",
        "num_batches",
        "loaded_local_tensors",
        "saved_local_tensors",
    ):
        mean_value = weighted_mean(key)
        if mean_value is not None:
            aggregated[f"{key}_mean"] = mean_value

    delta_values = [
        float(client_metrics["delta_norm_sq"])
        for _, client_metrics in metrics
        if "delta_norm_sq" in client_metrics
    ]
    if delta_values:
        aggregated["delta_norm_sq_max"] = max(delta_values)

    return aggregated

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
    if not hasattr(conf, "federated") or conf.federated is None:
        conf.federated = {}
    strategy_override = str(context.run_config.get("strategy-name", "") or "").strip()
    if strategy_override and strategy_override.lower() not in {"auto", "fedavg"}:
        conf.federated.server_strategy = strategy_override
    for run_key, cfg_key in (
        ("server-learning-rate", "server_learning_rate"),
        ("fedexp-min-lr", "fedexp_min_lr"),
        ("fedexp-max-lr", "fedexp_max_lr"),
        ("qffl-q", "qffl_q"),
        ("qffl-learning-rate", "qffl_learning_rate"),
        ("qffl-max-delta-multiplier", "qffl_max_delta_multiplier"),
        ("qffl-dynamic-step", "qffl_dynamic_step"),
        ("afl-lambda-lr", "afl_lambda_lr"),
        ("maxfl-loss-threshold", "maxfl_loss_threshold"),
        ("maxfl-temperature", "maxfl_temperature"),
    ):
        if run_key in context.run_config:
            conf.federated[cfg_key] = context.run_config[run_key]
    conf.model.prox_mu = resolve_prox_mu(conf, explicit_override=prox_mu_override)
    validation_summary = validate_runtime_config(conf)
    print(
        f"Starting server with l-ep={context.run_config['local-epochs']}, "
        f"ts={context.run_config['train-split']}, fclients={fraction_fit}, "
        f"prox_mu={conf.model.prox_mu}, fl={validation_summary['federated_method_preset']}"
        f", server_strategy={validation_summary['server_strategy']}"
    )
    conf.dataset.action_dim = infer_action_dim(conf)

    net_args = {
        "image_channels": get_expected_image_channels(conf),
        "low_dim_state_dim": infer_low_dim_state_dim(conf),
        "action_dim": int(conf.dataset.action_dim),
        "image_size": (128, 128),
        **get_agent_model_kwargs(conf),
    }
    print(f"Server model args: {net_args}")

    # Initialize model parameters
    agent = Agent(**net_args)
    trainable_manifest = get_trainable_parameter_manifest(agent, config=conf)
    print(
        "Server trainable surface: "
        f"tensors={trainable_manifest['num_trainable_tensors']} "
        f"params={trainable_manifest['trainable_params']} "
        f"local_only_params={trainable_manifest['local_only_params']} "
        f"fraction={trainable_manifest['trainable_fraction']:.6f}"
    )
    ndarrays = get_weights(agent, config=conf)
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

    strategy = TrainableOnlyFederatedStrategy(
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
    manifest_path = Path.joinpath(
        strategy.save_path,
        f"{strategy.save_name}.trainable_manifest.json",
    )
    manifest_path.write_text(
        json.dumps(trainable_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def get_server_app():
    return ServerApp(server_fn=server_fn)
# Create ServerApp
app = ServerApp(server_fn=server_fn)
