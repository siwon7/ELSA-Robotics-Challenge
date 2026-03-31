import argparse
import copy
import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from elsa_learning_agent.agent import Agent as LegacyBCAgent
from elsa_learning_agent.agent_forward_kinematics import (
    Agent,
    FrozenBackboneCLSExtractor,
    build_agent_kwargs,
    build_policy_kwargs_from_config,
    get_policy_class_name,
    infer_policy_name_from_model_path,
    policy_uses_cached_visual_features,
)
from elsa_learning_agent.dataset.compat import load_pickled_data
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.kinematics import LOW_DIM_STATE_DIM
from elsa_learning_agent.utils import (
    denormalize_action,
    get_image_transform,
    load_environment,
    process_obs,
)
from federated_elsa_robotics.task import validate_one_epoch


LEGACY_BC_POLICY_NAME = "legacy_bc"


def build_net_args(config=None, policy_name=None):
    if policy_name == LEGACY_BC_POLICY_NAME:
        return {
            "image_channels": 3,
            "low_dim_state_dim": 8,
            "action_dim": 8,
            "image_size": (128, 128),
        }

    kwargs = build_agent_kwargs(
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=8,
        image_size=(128, 128),
        config=config,
    )
    if policy_name is not None:
        kwargs["policy_name"] = policy_name
    return kwargs


def clone_config(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def infer_runtime_policy_name(model_path, config=None, policy_name=None):
    if policy_name is not None:
        return policy_name

    default_policy = build_policy_kwargs_from_config(config).get("policy_name")
    inferred = infer_policy_name_from_model_path(
        model_path,
        default=default_policy,
    )
    if os.path.basename(model_path).startswith("BCPolicy_"):
        return LEGACY_BC_POLICY_NAME
    return inferred


def load_agent(model_path, device, config=None, policy_name=None):
    policy_name = infer_runtime_policy_name(
        model_path,
        config=config,
        policy_name=policy_name,
    )

    if policy_name == LEGACY_BC_POLICY_NAME:
        agent = LegacyBCAgent(**build_net_args(config=config, policy_name=policy_name))
        agent.load_state_dict(model_path, device=device)
        agent.policy.to(device)
        agent.legacy_low_dim_state_dim = 8
        agent.eval()
        return agent

    agent = Agent(**build_net_args(config=config, policy_name=policy_name))
    state = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(state)
    agent.policy.to(device)
    agent.eval()
    return agent


def build_policy_input_adapter(agent, config, device):
    policy_name = getattr(agent, "policy_name", LEGACY_BC_POLICY_NAME)
    if not policy_uses_cached_visual_features(policy_name):
        return None

    policy_kwargs = build_policy_kwargs_from_config(config)
    extractor = FrozenBackboneCLSExtractor(
        backbone_name=policy_kwargs.get("backbone_name"),
        backbone_image_size=policy_kwargs.get("backbone_image_size"),
    ).to(device)
    extractor.eval()
    return extractor


def is_legacy_bc_agent(agent):
    return getattr(agent, "legacy_low_dim_state_dim", None) == 8


def legacy_bc_process_obs(obs, transform=None):
    front_image = torch.tensor(obs.front_rgb, dtype=torch.float32).permute(2, 0, 1) / 255
    if transform is not None:
        front_image = transform(front_image)

    low_dim_state = torch.tensor(
        np.concatenate((obs.joint_positions, np.array([obs.gripper_open]))),
        dtype=torch.float32,
    )
    return front_image, low_dim_state


class LegacyBCDataset(Dataset):
    def __init__(self, config, train=False, test=False, normalize=False):
        self.root_dir = config.dataset.root_dir
        task = config.dataset.task
        env_id = config.dataset.env_id
        train_split = config.dataset.train_split
        data_path = os.path.join(
            self.root_dir,
            task,
            f"env_{env_id}",
            "episodes_observations.pkl.gz",
        )

        demos_raw_data = load_pickled_data(data_path)
        self.normalize = normalize
        self.action_min = torch.tensor(config.transform.action_min)
        self.action_max = torch.tensor(config.transform.action_max)

        if train:
            demos_raw_data = demos_raw_data[: int(train_split * len(demos_raw_data))]
        elif test:
            demos_raw_data = demos_raw_data[int(config.dataset.test_split * len(demos_raw_data)) :]

        self.transform = get_image_transform(config)
        self.data = []

        print("Loading dataset from:", data_path)
        for demo in demos_raw_data:
            num_steps = len(demo) - 1
            for time_step in range(num_steps):
                self.data.append(self._load_datapoint(demo, time_step))

    def _load_datapoint(self, trajectory, time_step):
        obs = trajectory[time_step]
        next_obs = trajectory[time_step + 1]
        front_image, low_dim_state = legacy_bc_process_obs(obs, self.transform)
        action = torch.tensor(
            np.concatenate((obs.joint_velocities, np.array([next_obs.gripper_open]))),
            dtype=torch.float32,
        )
        if self.normalize:
            action = 2 * ((action - self.action_min) / (self.action_max - self.action_min)) - 1
        return {
            "action": action,
            "low_dim_state": low_dim_state,
            "image": front_image,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def legacy_bc_load_environment(base_cfg, collection_cfg, idx_environment, headless=True):
    from colosseum import TASKS_PY_FOLDER, TASKS_TTM_FOLDER
    from colosseum.rlbench.extensions.environment import EnvironmentExt
    from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete

    task = name_to_class(base_cfg.env.task_name, TASKS_PY_FOLDER)

    config = clone_config(base_cfg)
    env_entries = collection_cfg.get("env_config", [])
    env_entry = next(
        (entry for entry in env_entries if entry.get("env_idx") == idx_environment),
        None,
    )
    if env_entry is None:
        raise ValueError(f"Environment index {idx_environment} not found in collection config")

    env_factors = config.env.scene.factors
    for variation_cfg in env_entry.get("variations_parameters", []):
        var_type = variation_cfg["type"]
        var_name = variation_cfg.get("name")
        for factor_cfg in env_factors:
            if factor_cfg.variation != var_type:
                continue
            if var_name is not None and "name" in factor_cfg and factor_cfg.name != var_name:
                continue
            for key, value in variation_cfg.items():
                if key == "type":
                    continue
                factor_cfg[key] = value
            break

    data_cfg, env_cfg = config.data, config.env

    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        ),
        obs_config=ObservationConfigExt(data_cfg),
        headless=headless,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=env_cfg,
    )
    rlbench_env.launch()
    task_env = rlbench_env.get_task(task)
    return task_env, rlbench_env


def build_split_config(base_config, task, split, env_id):
    config = clone_config(base_config)
    config.dataset.task = task
    config.dataset.env_id = env_id
    if "train_split" not in config.dataset:
        config.dataset.train_split = 1.0
    if "test_split" not in config.dataset:
        config.dataset.test_split = 0.0
    if split == "training":
        config.dataset.root_dir = config.dataset.root_dir
    elif split == "eval":
        config.dataset.root_dir = config.dataset.root_eval_dir
    elif split == "test":
        config.dataset.root_dir = config.dataset.root_test_dir
    else:
        raise ValueError(f"Unsupported split: {split}")
    return config


def build_offline_env_ids(config, split):
    if split == "eval":
        start, end = config.dataset.final_eval_env_idx_range
    elif split == "test":
        start, end = config.dataset.final_test_env_idx_range
    else:
        raise ValueError(f"Unsupported split: {split}")
    return list(range(start, end))


def build_live_env_ids(config, split):
    if split == "eval":
        return list(config.dataset.final_eval_live_idxs)
    if split == "test":
        return list(config.dataset.final_test_live_idxs)
    raise ValueError(f"Unsupported split: {split}")


def evaluate_offline(agent, base_config, task, split, device, batch_size, num_workers):
    env_ids = build_offline_env_ids(base_config, split)
    loss_per_env = {}
    losses = []
    dataset_cls = LegacyBCDataset if is_legacy_bc_agent(agent) else ImitationDataset
    for env_id in env_ids:
        config = build_split_config(base_config, task, split, env_id)
        loader = DataLoader(
            dataset_cls(config=config, train=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        loss = validate_one_epoch(agent, loader, device=device)
        loss_per_env[env_id] = float(loss)
        losses.append(float(loss))
    return {
        "env_ids": env_ids,
        "loss_per_env": loss_per_env,
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
    }


def run_live_episodes(
    agent,
    device,
    transform,
    base_cfg,
    idx_environment,
    num_episodes,
    headless,
):
    collection_cfg_path = os.path.join(
        base_cfg.dataset.root_dir,
        base_cfg.env.task_name,
        f"{base_cfg.env.task_name}_fed.json",
    )
    with open(collection_cfg_path, "r", encoding="utf-8") as fh:
        collection_cfg = json.load(fh)

    if is_legacy_bc_agent(agent):
        task_env, rlbench_env = legacy_bc_load_environment(
            base_cfg,
            collection_cfg,
            idx_environment,
            headless=headless,
        )
    else:
        task_env, rlbench_env = load_environment(
            base_cfg,
            collection_cfg,
            idx_environment,
            headless=headless,
        )
    input_adapter = build_policy_input_adapter(agent, base_cfg, device)
    rewards = []
    max_steps = int(os.environ.get("ELSA_SIM_MAX_STEPS", "300"))
    try:
        for _ in range(num_episodes):
            _, obs = task_env.reset()
            terminate = False
            reward = 0.0
            steps = 0
            while not terminate and steps < max_steps:
                with torch.no_grad():
                    if is_legacy_bc_agent(agent):
                        front_rgb, low_dim_state = legacy_bc_process_obs(obs, transform)
                    else:
                        front_rgb, low_dim_state = process_obs(obs, transform)
                    front_rgb = front_rgb.unsqueeze(0).to(device)
                    if input_adapter is not None:
                        # Convert inference-mode extractor output into a regular tensor
                        # before passing it through the trainable policy head.
                        front_rgb = input_adapter(front_rgb).float().clone()
                    low_dim_state = low_dim_state.unsqueeze(0).to(device)
                    action = agent.get_action(front_rgb, low_dim_state)
                denormalized_action = denormalize_action(
                    action.detach().cpu(),
                    torch.tensor(base_cfg.transform.action_min),
                    torch.tensor(base_cfg.transform.action_max),
                )
                obs, reward, terminate = task_env.step(denormalized_action.numpy()[0])
                steps += 1
            rewards.append(float(reward))
    finally:
        rlbench_env.shutdown()
    return rewards


def evaluate_online(agent, base_config, task, split, device):
    live_cfg = clone_config(base_config)
    if split == "eval":
        live_cfg.dataset.root_dir = live_cfg.dataset.root_eval_dir
    elif split == "test":
        live_cfg.dataset.root_dir = live_cfg.dataset.root_test_dir
    else:
        raise ValueError(f"Unsupported split: {split}")

    fed_cfg = OmegaConf.load(os.path.join(live_cfg.dataset.root_dir, task, f"{task}_fed.yaml"))
    live_cfg.env = fed_cfg.env
    live_cfg.data = fed_cfg.data
    live_cfg.transform = clone_config(base_config).transform
    live_cfg.data.renderer = os.environ.get("ELSA_SIM_RENDERER", "opengl")
    transform = get_image_transform(base_config)

    env_ids = build_live_env_ids(base_config, split)
    max_envs = os.environ.get("ELSA_SIM_MAX_ENVS")
    if max_envs:
        env_ids = env_ids[: int(max_envs)]
    num_episodes = int(
        os.environ.get("ELSA_SIM_NUM_EPISODES", base_config.dataset.num_episodes_live)
    )
    headless = os.environ.get("ELSA_SIM_HEADLESS", "0") == "1"
    rewards_per_env = {}
    flattened_rewards = []
    for env_id in env_ids:
        rewards = run_live_episodes(
            agent=agent,
            device=device,
            transform=transform,
            base_cfg=live_cfg,
            idx_environment=env_id,
            num_episodes=num_episodes,
            headless=headless,
        )
        rewards_per_env[env_id] = rewards
        flattened_rewards.extend(rewards)

    return {
        "env_ids": env_ids,
        "rewards_per_env": rewards_per_env,
        "mean_reward": float(np.mean(flattened_rewards)),
        "std_reward": float(np.std(flattened_rewards)),
    }


def checkpoint_stem(local_epochs, fraction_fit, train_test_split, strategy_name, policy_name):
    return (
        f"{strategy_name}_{get_policy_class_name(policy_name)}"
        f"_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}"
    )


def checkpoint_pattern(task, local_epochs, fraction_fit, train_test_split, strategy_name, policy_name):
    return os.path.join(
        "model_checkpoints",
        task,
        checkpoint_stem(
            local_epochs=local_epochs,
            fraction_fit=fraction_fit,
            train_test_split=train_test_split,
            strategy_name=strategy_name,
            policy_name=policy_name,
        )
        + "_round_*.pth",
    )


def discover_checkpoints(task, local_epochs, fraction_fit, train_test_split, strategy_name, policy_name):
    round_to_path = {}
    for path in glob.glob(
        checkpoint_pattern(
            task,
            local_epochs,
            fraction_fit,
            train_test_split,
            strategy_name,
            policy_name,
        )
    ):
        round_str = os.path.splitext(path)[0].rsplit("_round_", 1)[-1]
        if round_str.isdigit():
            round_to_path[int(round_str)] = path
    return round_to_path


def ensure_results_dir(task, local_epochs, fraction_fit, train_test_split, strategy_name, policy_name):
    path = os.path.join(
        "results",
        task,
        checkpoint_stem(
            local_epochs=local_epochs,
            fraction_fit=fraction_fit,
            train_test_split=train_test_split,
            strategy_name=strategy_name,
            policy_name=policy_name,
        ),
    )
    os.makedirs(path, exist_ok=True)
    return path


def save_outputs(results, result_dir, plotting):
    results_json = os.path.join(result_dir, "results.json")
    with open(results_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    results_txt = os.path.join(result_dir, "results.txt")
    with open(results_txt, "w", encoding="utf-8") as fh:
        for result in results:
            fh.write(
                f"round={result['round']} "
                f"mean_loss={result['offline']['mean_loss']:.6f} "
                f"std_loss={result['offline']['std_loss']:.6f} "
                f"mean_success={result.get('online', {}).get('mean_reward', 0.0):.6f} "
                f"std_success={result.get('online', {}).get('std_reward', 0.0):.6f}\n"
            )

    if plotting and results:
        rounds = [result["round"] for result in results]
        mean_losses = [result["offline"]["mean_loss"] for result in results]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mean_losses, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Mean Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "mean_loss.png"))
        plt.close()

        if "online" in results[0]:
            mean_rewards = [result["online"]["mean_reward"] for result in results]
            plt.figure(figsize=(10, 5))
            plt.plot(rounds, mean_rewards, marker="o")
            plt.xlabel("Round")
            plt.ylabel("Mean Success Rate")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "mean_success_rate.png"))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ELSA checkpoints offline and online.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--local_epochs", type=int, default=50)
    parser.add_argument("--fraction_fit", type=float, default=0.05)
    parser.add_argument("--train_test_split", type=float, default=0.9)
    parser.add_argument("--strategy_name", type=str, default="fedavg")
    parser.add_argument("--policy_name", type=str, default=None)
    parser.add_argument("--round", type=int, nargs="*", default=None)
    parser.add_argument("--rounds_to_evaluate", type=int, default=None)
    parser.add_argument("--split", type=str, default="eval", choices=["eval", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    config = OmegaConf.load("dataset_config.yaml")
    policy_name = args.policy_name or build_policy_kwargs_from_config(config).get(
        "policy_name"
    )
    checkpoint_map = discover_checkpoints(
        task=args.task,
        local_epochs=args.local_epochs,
        fraction_fit=args.fraction_fit,
        train_test_split=args.train_test_split,
        strategy_name=args.strategy_name,
        policy_name=policy_name,
    )
    if not checkpoint_map:
        raise FileNotFoundError("No checkpoints found for the requested configuration.")

    if args.round:
        rounds = [round_num for round_num in args.round if round_num in checkpoint_map]
    else:
        rounds = sorted(checkpoint_map)
        if args.rounds_to_evaluate is not None:
            rounds = rounds[: args.rounds_to_evaluate]
    if not rounds:
        raise FileNotFoundError("Requested rounds do not exist.")

    device = torch.device(args.device)
    results = []
    for round_num in rounds:
        model_path = checkpoint_map[round_num]
        print(f"Evaluating round {round_num}: {model_path}")
        agent = load_agent(
            model_path,
            device=device,
            config=config,
            policy_name=policy_name,
        )
        offline = evaluate_offline(
            agent=agent,
            base_config=config,
            task=args.task,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        result = {"round": round_num, "model_path": model_path, "offline": offline}
        if args.simulator:
            result["online"] = evaluate_online(
                agent=agent,
                base_config=config,
                task=args.task,
                split=args.split,
                device=args.device,
            )
        results.append(result)

    result_dir = ensure_results_dir(
        task=args.task,
        local_epochs=args.local_epochs,
        fraction_fit=args.fraction_fit,
        train_test_split=args.train_test_split,
        strategy_name=args.strategy_name,
        policy_name=policy_name,
    )
    save_outputs(results, result_dir, plotting=args.plotting)

    best_key = (
        lambda item: item["online"]["mean_reward"]
        if "online" in item
        else -item["offline"]["mean_loss"]
    )
    best_result = max(results, key=best_key)
    print(json.dumps(best_result, indent=2))
    print(f"Saved results to {result_dir}")
    print(f"Total evaluation time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
