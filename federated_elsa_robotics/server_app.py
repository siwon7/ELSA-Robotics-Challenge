"""elsa-robotics: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader
from federated_elsa_robotics.policy_runtime import (
    DEFAULT_SEED,
    build_runtime_agent,
    get_runtime_policy_name,
    parse_bool,
    resolve_run_config,
    set_global_seed,
)
from federated_elsa_robotics.eval_model import LEGACY_BC_POLICY_NAME, LegacyBCDataset
from federated_elsa_robotics.strategies import build_strategy
from federated_elsa_robotics.task import get_weights, set_weights, validate_one_epoch
from omegaconf import OmegaConf
import torch

from elsa_learning_agent.dataset.compat import get_action_dim
from elsa_learning_agent.kinematics import LOW_DIM_STATE_DIM
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset

def gen_evaluate_fn(
    testloader: list[DataLoader],
    device: torch.device,
    net_args: dict,
    simulator: bool = False,
    dataset_config: dict = None,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        agent = build_runtime_agent(**net_args)
        set_weights(agent, parameters_ndarrays)
        agent.policy.to(device)

        metrics = {}

        total_loss = 0.0
        metrics["loss_per_env"] = {}
        for idx, dataset in enumerate(testloader):
            loss = validate_one_epoch(agent, dataset, device=device)
            total_loss += loss
            metrics["loss_per_env"][dataset_config.dataset["test_env_idx_range"][0] + idx] = loss
        
        loss = total_loss / len(testloader)

        if simulator:
            from federated_elsa_robotics.eval_model import online_evaluation
            from elsa_learning_agent.utils import get_image_transform

            # Load base config from dataset yaml
            base_cfg = OmegaConf.load(dataset_config.dataset["root_dir"] + f"/{dataset_config.dataset['task']}/{dataset_config.dataset['task']}_fed.yaml")
            base_cfg.dataset = dataset_config.dataset
            base_cfg.transform = dataset_config.transform

            metrics["avg_reward"] = 0.0
            metrics["video_array"] = []
            for idx_evaluate in dataset_config.dataset["test_live_idxs"]:                
                # Perform online evaluation
                avg_reward, video_array = online_evaluation(agent, device, get_image_transform(dataset_config), base_cfg, idx_evaluate, dataset_config.dataset["num_episodes_live"])
                metrics["avg_reward"] += avg_reward
                metrics["video_array"].append(video_array)
            metrics["avg_reward"] /= len(dataset_config.dataset["test_live_idxs"])
            
        return loss, metrics

    return evaluate


def train_aggregation_fn(metrics: list[dict]):
    """Aggregate training metrics."""
    losses = [num_samples * m["train_loss"] for num_samples, m in metrics]
    examples = [num_samples for num_samples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}

def server_fn(context: Context):
    # Read from project toml config
    run_config = resolve_run_config(context.run_config)
    num_rounds = run_config["num-server-rounds"]
    fraction_fit = run_config["fraction-fit"]
    fraction_evaluate = run_config["fraction-eval"]
    server_device = run_config["server-device"]
    use_wandb = run_config["use-wandb"]
    enable_centralized_eval = run_config.get("enable-centralized-eval", False)
    centralized_eval_simulator = run_config.get("centralized-eval-simulator", False)
    centralized_eval_batch_size = int(run_config.get("centralized-eval-batch-size", 32))
    centralized_eval_num_workers = int(run_config.get("centralized-eval-num-workers", 8))
    dataset_config_path = run_config["dataset-config-path"]
    strategy_name = run_config["strategy-name"]
    conf = OmegaConf.load(dataset_config_path)
    policy_name = get_runtime_policy_name(run_config, conf)
    conf.model.policy_name = policy_name
    seed = int(run_config.get("seed", DEFAULT_SEED))
    deterministic = parse_bool(run_config.get("deterministic-training", False))
    set_global_seed(seed, deterministic)
    print(
        f"Starting server with strategy={strategy_name}, "
        f"policy={policy_name}, "
        f"l-ep={run_config['local-epochs']}, "
        f"ts={run_config['train-split']}, fclients={fraction_fit}"
    )

    net_args = {
        "policy_name": policy_name,
        "image_size": (128, 128),
        "action_dim": get_action_dim(conf),
        "low_dim_state_dim": LOW_DIM_STATE_DIM,
        "config": conf,
    }

    # Initialize model parameters
    agent = build_runtime_agent(**net_args)
    ndarrays = get_weights(agent)
    parameters = ndarrays_to_parameters(ndarrays)

    # Evaluation loader
    def create_config(idx): 
        cur_config = conf.copy()
        cur_config.dataset.task = run_config["dataset-task"]
        cur_config.dataset.env_id = idx
        # Use evaluation dataset for the server
        cur_config.dataset.root_dir = cur_config.dataset.root_eval_dir
        cur_config.dataset.test_split = 0.0
        cur_config.dataset.train_split = run_config["train-split"]
        cur_config.dataset.num_server_rounds = num_rounds
        cur_config.dataset.local_epochs = run_config["local-epochs"]
        cur_config.dataset.save_rounds = run_config.get("save-rounds", "5,25,50,100")
        return cur_config
    config = create_config(0)
    test_dataset = None
    evaluate_fn = None
    if enable_centralized_eval:
        dataset_cls = (
            LegacyBCDataset if policy_name == LEGACY_BC_POLICY_NAME else ImitationDataset
        )
        test_dataset = [
            DataLoader(
                dataset_cls(config=create_config(idx), test=True),
                batch_size=centralized_eval_batch_size,
                shuffle=False,
                num_workers=centralized_eval_num_workers,
            )
            for idx in range(*config.dataset["test_env_idx_range"])
        ]
        evaluate_fn = gen_evaluate_fn(
            test_dataset,
            server_device,
            net_args,
            simulator=centralized_eval_simulator,
            dataset_config=config,
        )
    
    # Define strategy
    strategy = build_strategy(
        strategy_name,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=parameters,
        agent=agent,
        config=config,
        use_wandb=use_wandb,
        evaluate_fn=evaluate_fn,
        fit_aggregation_fn=train_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def get_server_app():
    return ServerApp(server_fn=server_fn)
# Create ServerApp
app = ServerApp(server_fn=server_fn)
