import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from elsa_learning_agent.utils import get_image_transform
from elsa_learning_agent.dataset.compat import load_pickled_data


class EvalImitationDataset(Dataset):
    def __init__(self, config, train=False, test=False):
        self.root_dir = config.dataset.root_eval_dir
        task = config.dataset.task
        env_id = config.dataset.env_id
        data_path = os.path.join(self.root_dir, f"{task}", f"env_{env_id}", "episodes_observations.pkl.gz")
        
        obs_raw_data = load_pickled_data(data_path)

        self.transform = get_image_transform(config)
        self.data = []
        self.demos_idx = []

        # Load data
        print("Loading dataset from:", data_path)
        for key, obs in tqdm(obs_raw_data.items()):
            self.data.append(self._load_datapoint(obs))

    def _load_datapoint(self, obs):
        front_image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1)/255
        front_image = self.transform(front_image)
        low_dim_state = torch.tensor(obs["low_dim_state"], dtype=torch.float32)
        obs_id = obs["obs_id"]

        return {
            "obs_id": obs_id,
            "low_dim_state": low_dim_state,
            "image": front_image,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
