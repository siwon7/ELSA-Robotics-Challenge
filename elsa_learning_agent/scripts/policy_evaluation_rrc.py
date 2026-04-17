import signal
import sys
import os
import torch

from elsa_learning_agent.agent import Agent
from elsa_learning_agent.config_utils import load_runtime_config
from elsa_learning_agent.dataset.dataset_loader_eval import EvalImitationDataset
from elsa_learning_agent.utils import (
    denormalize_action,
    get_action_output_activation,
    get_execution_action_adapter,
    get_execution_action_interface,
    get_joint_velocity_servo_clip,
    get_joint_velocity_servo_gain,
    select_receding_horizon_action,
)
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

from federated_elsa_robotics.task import infer_action_dim


def _adapt_to_submission_action(predicted_action, current_low_dim_state, cfg):
    action = select_receding_horizon_action(predicted_action, cfg)
    action = denormalize_action(
        action.detach().cpu(),
        torch.tensor(cfg.transform.action_min),
        torch.tensor(cfg.transform.action_max),
    )

    execution_interface = str(get_execution_action_interface(cfg))
    execution_adapter = str(get_execution_action_adapter(cfg))
    if (
        execution_interface == "joint_velocity"
        and execution_adapter == "joint_position_to_joint_velocity_servo"
    ):
        target_joint_positions = action[:7]
        current_joint_positions = current_low_dim_state[:7].detach().cpu()
        servo_gain = float(get_joint_velocity_servo_gain(cfg))
        servo_clip = float(get_joint_velocity_servo_clip(cfg))
        joint_velocity = torch.clamp(
            servo_gain * (target_joint_positions - current_joint_positions),
            min=-servo_clip,
            max=servo_clip,
        )
        return torch.cat((joint_velocity, action[7:8]), dim=0)
    return action


def validate_one_epoch(agent, val_loader, cfg, task, device):
    agent.eval()
    predicted_actions= {f"{task}": []}
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch["image"].to(device)
            low_dim_state = batch["low_dim_state"].to(device)
            obs_ids = batch["obs_id"]

            predicted_action = agent.get_action(image, low_dim_state)

            for idx, action in enumerate(predicted_action):
                submission_action = _adapt_to_submission_action(
                    action,
                    low_dim_state[idx],
                    cfg,
                )
                datapoint = {"action_id": int(obs_ids[idx]),  "action": submission_action.numpy().tolist()}
                predicted_actions[f"{task}"].append(datapoint)


    return predicted_actions


def main(model_paths, model_config_path, predictions_path) -> int:

    # load json containin all default predictions
    all_predictions = {"data": {}}

    for task in ["slide_block_to_target", "close_box", "scoop_with_spatula", "insert_onto_square_peg"]:
        model_path = model_paths[task]
        if model_path is None:
            print("No model path provided for task: ", task) 
            print("Please provide the weights for task: ", task) 
            print("Closing the script...")     
            os.kill(os.getppid(),signal.SIGTERM)
            sys.exit(2)
        else:

            model_cfg = load_runtime_config(model_config_path, task=task)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_cfg.dataset.root_eval_dir = model_cfg.dataset.root_test_dir

            dataset = EvalImitationDataset(model_cfg, train=False, test=True)
            # Important not to shuffle the data
            dataloader = DataLoader(
                dataset,
                batch_size=model_cfg.dataset.batch_size,
                shuffle=False,
                num_workers=int(getattr(model_cfg.dataset, "num_workers", 0) or 0),
            )

            agent = Agent(
                image_channels=3,
                low_dim_state_dim=8,
                action_dim=int(infer_action_dim(model_cfg)),
                image_size=(128,128),
                vision_backbone=str(getattr(model_cfg.model, "vision_backbone", "cnn")),
                projector_dim=int(getattr(model_cfg.model, "projector_dim", 256)),
                action_output_activation=get_action_output_activation(model_cfg),
                normalize_branch_embeddings=bool(
                    getattr(model_cfg.model, "normalize_branch_embeddings", False)
                ),
                low_dim_dropout_prob=float(
                    getattr(model_cfg.model, "low_dim_dropout_prob", 0.0) or 0.0
                ),
            )
            agent.policy.to(device)

            print(f"Loading model from: {model_path}")
            agent.load_state_dict(model_path)

            print("Starting evaluation...")
            predctions = validate_one_epoch(agent, dataloader, model_cfg, task, device)

            all_predictions["data"][task] = predctions[task]

    # find the dir of the default predictions
    save_action_path =  os.path.join(os.path.dirname(predictions_path),  "actions.json")

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_action_path), exist_ok=True)

    with open(save_action_path, "w") as f:
        json.dump(all_predictions, f, indent=4)
    
    print(f"Saved actions dataset to: {save_action_path}")


if __name__ == "__main__":
    # TODO: set the correct model paths with your best trained model for each task
    model_paths = {
        "slide_block_to_target": "./model_checkpoints/slide_block_to_target/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth", 
        "close_box": "./model_checkpoints/close_box/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth", 
        "scoop_with_spatula": "./model_checkpoints/scoop_with_spatula/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth",  
        "insert_onto_square_peg": "./model_checkpoints/insert_onto_square_peg/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth", 
        }
    
    
    model_config_path = "./dataset_config.yaml"
    
    # Set the path to save the predictions
    predictions_path = "./results/predictions.json"
    main(model_paths, model_config_path, predictions_path)
