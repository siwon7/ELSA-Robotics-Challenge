import gzip
import pickle

import numpy as np
import torch
from torchvision import transforms
from elsa_learning_agent.kinematics import build_low_dim_state


class CompatObservation:
    """Lightweight stand-in for RLBench observations used during unpickling."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class CompatDataContainer:
    """Compatibility wrapper matching the pickled dataset container layout."""

    def __init__(self):
        self.data = None

    def __setstate__(self, state):
        self.__dict__.update(state)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "colosseum.rlbench.datacontainer" and name == "DataContainer":
            return CompatDataContainer
        if module == "rlbench.backend.observation" and name == "Observation":
            return CompatObservation
        return super().find_class(module, name)


def load_pickled_data(path):
    with gzip.open(path, "rb") as file_obj:
        container = CompatUnpickler(file_obj).load()

    return container.data if hasattr(container, "data") else container


def get_image_transform(config_file):
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=config_file.transform.normalize_mean,
                std=config_file.transform.normalize_std,
            )
        ]
    )


def process_obs(obs, transform=None):
    front_image = torch.tensor(obs.front_rgb, dtype=torch.float32).permute(2, 0, 1) / 255
    if transform is not None:
        front_image = transform(front_image)

    low_dim_state = torch.tensor(
        build_low_dim_state(obs.joint_positions, obs.gripper_open),
        dtype=torch.float32,
    )
    return front_image, low_dim_state


def normalize_action(action, action_min, action_max):
    return 2 * ((action - action_min) / (action_max - action_min)) - 1
