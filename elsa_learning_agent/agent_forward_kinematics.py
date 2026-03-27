import torch
import torch.nn as nn

from elsa_learning_agent.kinematics import (
    EE_FEATURE_DIM,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    LOW_DIM_STATE_DIM,
    NUM_ARM_JOINTS,
)


class Agent:
    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
    ):
        self.policy = FKBCPolicy(
            image_channels=image_channels,
            low_dim_state_dim=low_dim_state_dim,
            action_dim=action_dim,
            image_size=image_size,
        )

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def get_action(self, image, low_dim_state):
        return self.policy(image, low_dim_state)

    def load_state_dict(self, state_dict, device=None):
        if device is None:
            self.policy.load_state_dict(torch.load(state_dict))
        else:
            self.policy.load_state_dict(
                torch.load(state_dict, map_location=torch.device(device))
            )
        return self

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        return self


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=256, image_size=(64, 64)):
        super().__init__()
        self.image_size = image_size

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.flattened_size = self._calculate_flattened_size(input_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, output_dim),
            nn.ReLU(),
        )

    def _calculate_flattened_size(self, input_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, input_channels, self.image_size[0], self.image_size[1]
            )
            dummy_output = self.cnn(dummy_input)
            return int(torch.flatten(dummy_output, start_dim=1).size(1))

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class FKBCPolicy(nn.Module):
    """BC policy with separate embeddings for joints, FK end-effector, and gripper."""

    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
    ):
        super().__init__()
        expected_dim = NUM_ARM_JOINTS + EE_FEATURE_DIM + 1
        if low_dim_state_dim != expected_dim:
            raise ValueError(
                f"FKBCPolicy expects low_dim_state_dim={expected_dim}, got {low_dim_state_dim}"
            )

        self.cnn_encoder = CNNEncoder(
            input_channels=image_channels,
            output_dim=256,
            image_size=image_size,
        )
        self.joint_encoder = MLPEncoder(input_dim=NUM_ARM_JOINTS, hidden_dim=128, output_dim=128)
        self.ee_pos_encoder = MLPEncoder(
            input_dim=EE_POS_DIM, hidden_dim=64, output_dim=64
        )
        self.ee_rot_encoder = MLPEncoder(
            input_dim=EE_ROT6D_DIM, hidden_dim=128, output_dim=64
        )
        self.gripper_encoder = MLPEncoder(input_dim=1, hidden_dim=32, output_dim=32)
        self.mlp_policy = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 64 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def _split_low_dim_state(self, low_dim_state):
        joint_state = low_dim_state[..., :NUM_ARM_JOINTS]
        ee_pos_state = low_dim_state[
            ..., NUM_ARM_JOINTS : NUM_ARM_JOINTS + EE_POS_DIM
        ]
        ee_rot_state = low_dim_state[
            ...,
            NUM_ARM_JOINTS + EE_POS_DIM : NUM_ARM_JOINTS + EE_FEATURE_DIM,
        ]
        gripper_state = low_dim_state[..., -1:].contiguous()
        return joint_state, ee_pos_state, ee_rot_state, gripper_state

    def forward(self, image, low_dim_state):
        joint_state, ee_pos_state, ee_rot_state, gripper_state = self._split_low_dim_state(low_dim_state)

        img_embedding = self.cnn_encoder(image)
        joint_embedding = self.joint_encoder(joint_state)
        ee_pos_embedding = self.ee_pos_encoder(ee_pos_state)
        ee_rot_embedding = self.ee_rot_encoder(ee_rot_state)
        gripper_embedding = self.gripper_encoder(gripper_state)

        fused = torch.cat(
            [
                img_embedding,
                joint_embedding,
                ee_pos_embedding,
                ee_rot_embedding,
                gripper_embedding,
            ],
            dim=-1,
        )
        return self.mlp_policy(fused)
