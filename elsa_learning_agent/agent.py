####################################
# This file is part of ELSA.
# 
# In this file, you need to implement an **agent** that will be used to interact with the environment. 
# The agent should be responsible for processing observations and selecting actions.
#
# **Agent Requirements:**
# The agent must be implemented as a class that inherits from `torch.nn.Module` and should include:
#
# - `__init__(self)`:  
#   - Define the architecture of your neural network here.  
#   - You can use convolutional layers for image processing and fully connected layers for decision-making.
#
# - `get_action(self, image, low_dim_state)`:  
#   - This method takes an image and a low-dimensional state as input and returns an action.
#   - You can apply post-processing (e.g., argmax for discrete actions or squashing for continuous actions).
#
#
# **Expected Input and Output for the get_action function:**
# - Inputs:
#   - `image`: A tensor representing the visual observation (e.g., RGB image).
#   - `low_dim_state`: A tensor containing additional state information.
# - Output:
#   - An action tensor that determines the agent's response to the given input. 
#    NOTE: The output action should be between [-1,1] ad it will be later denormalized to the action space of the environment.
#
# **Example Implementation:**
# Below is an example of a behavioral cloning model that combines a CNN encoder for image processing 
# and an MLP encoder for low-dimensional state inputs. It outputs an action based on the processed observations.
#
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from elsa_learning_agent.model_registry import get_supported_vision_backbones

class Agent():
    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=10,
        action_dim=4,
        image_size=(64, 64),
        vision_backbone="cnn",
        projector_dim=256,
        action_output_activation="tanh",
        normalize_branch_embeddings=False,
        low_dim_dropout_prob=0.0,
    ):
        # YOUR CODE HERE
        # Define the architecture of your neural network here
        # We provide an example by implementung a CNN policy we called BC Policy
        self.policy = BCPolicy(
        image_channels=image_channels,
        low_dim_state_dim=low_dim_state_dim,
        action_dim=action_dim,
        image_size=image_size,
        vision_backbone=vision_backbone,
        projector_dim=projector_dim,
        action_output_activation=action_output_activation,
        normalize_branch_embeddings=normalize_branch_embeddings,
        low_dim_dropout_prob=low_dim_dropout_prob,
        )

    def train(self, ):
        self.policy.train()

    def eval(self, ):
        self.policy.eval()

    def get_action(self, image, low_dim_state):
        # Get the action from the policy
        return self.policy(image, low_dim_state) # Assuming the policy is a function
    
    def load_state_dict(self, state_dict,device=None):
        if device is None:
            self.policy.load_state_dict(torch.load(state_dict))
        else:
            self.policy.load_state_dict(torch.load(state_dict), map_location=torch.device(device))
        return self
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        return self



####################################################
# Your code here
# ...
#
#
#
######## EXAMPLE IMPLEMENTATION of CNN policy ########

class CNNEncoder(nn.Module):
    """CNN to encode the image observation with dynamic adjustment for fully connected layers."""
    def __init__(self, input_channels=3, output_dim=256, image_size=(64, 64)):
        """
        Args:
            input_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            output_dim (int): Size of the output embedding.
            image_size (tuple): Size of the input image (height, width).
        """
        super(CNNEncoder, self).__init__()
        self.image_size = image_size
        self.output_dim = output_dim

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Dynamically calculate the flattened size after convolutions
        self.flattened_size = self._calculate_flattened_size()

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, output_dim),
            nn.ReLU()
        )

    def _calculate_flattened_size(self):
        """Calculate the size of the flattened feature map after convolutions."""
        with torch.no_grad():
            # Create a dummy tensor with the given image size
            dummy_input = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
            dummy_output = self.cnn(dummy_input)  # Pass it through the CNN
            return int(torch.flatten(dummy_output, start_dim=1).size(1))  # Flatten and get the size

    def forward(self, x):
        x = self.cnn(x)  # Apply convolutions
        x = torch.flatten(x, start_dim=1)  # Flatten the feature map
        x = self.fc(x)  # Apply the fully connected layer
        return x


class FrozenDinoV3Encoder(nn.Module):
    """Frozen DINOv3-S/16 image encoder with a small trainable projector."""

    def __init__(self, input_channels=3, output_dim=256):
        super(FrozenDinoV3Encoder, self).__init__()
        if input_channels != 3:
            raise ValueError(
                "dinov3_vits16_frozen expects 3-channel RGB input, "
                f"got input_channels={input_channels}."
            )

        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as exc:
            raise ImportError(
                "timm is required when vision_backbone='dinov3_vits16_frozen'."
            ) from exc

        self.backbone = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=True,
            num_classes=0,
        )
        self.output_dim = output_dim
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.num_features, output_dim),
            nn.ReLU(),
        )
        data_cfg = resolve_model_data_config(self.backbone)
        self.register_buffer(
            "backbone_mean",
            torch.tensor(data_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "backbone_std",
            torch.tensor(data_cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        x = ((x * 0.5) + 0.5).clamp(0.0, 1.0)
        x = (x - self.backbone_mean) / self.backbone_std
        with torch.no_grad():
            features = self.backbone(x)
        return self.projector(features)


class FrozenDepthAnythingEncoder(nn.Module):
    """Frozen Depth Anything encoder with a lightweight trainable depth projector."""

    def __init__(
        self,
        input_channels=3,
        output_dim=256,
        model_id="LiheYoung/depth-anything-small-hf",
    ):
        super(FrozenDepthAnythingEncoder, self).__init__()
        if input_channels != 3:
            raise ValueError(
                "depth_anything_small_frozen expects 3-channel RGB input, "
                f"got input_channels={input_channels}."
            )
        try:
            from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "transformers is required when vision_backbone uses Depth Anything."
            ) from exc

        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.backbone = DepthAnythingForDepthEstimation.from_pretrained(model_id)
        self.output_dim = output_dim
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(16 * 16, output_dim),
            nn.ReLU(),
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        image_batch = ((x * 0.5) + 0.5).clamp(0.0, 1.0)
        images = [
            sample.detach().permute(1, 2, 0).cpu().numpy()
            for sample in image_batch
        ]
        encoded = self.image_processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(x.device)
        with torch.no_grad():
            depth = self.backbone(pixel_values).predicted_depth.unsqueeze(1)
        depth_min = depth.amin(dim=(-2, -1), keepdim=True)
        depth_max = depth.amax(dim=(-2, -1), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        return self.projector(depth)


class FrozenDinoDepthConcatEncoder(nn.Module):
    """Concatenate frozen DINOv3 and Depth Anything features."""

    def __init__(self, input_channels=3, output_dim=256):
        super(FrozenDinoDepthConcatEncoder, self).__init__()
        dino_dim = output_dim // 2
        depth_dim = output_dim - dino_dim
        self.dino_encoder = FrozenDinoV3Encoder(
            input_channels=input_channels,
            output_dim=dino_dim,
        )
        self.depth_encoder = FrozenDepthAnythingEncoder(
            input_channels=input_channels,
            output_dim=depth_dim,
        )

    def train(self, mode=True):
        super().train(mode)
        self.dino_encoder.train(mode)
        self.depth_encoder.train(mode)
        return self

    def forward(self, x):
        dino_features = self.dino_encoder(x)
        depth_features = self.depth_encoder(x)
        return torch.cat([dino_features, depth_features], dim=-1)

class MLPEncoder(nn.Module):
    """MLP to encode the low-dimensional state."""
    def __init__(self, input_dim, output_dim=128):
        super(MLPEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)
    

class BCPolicy(nn.Module):
    """Behavior Cloning model combining CNN and MLP encoders."""
    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=10,
        action_dim=4,
        image_size=(64, 64),
        vision_backbone="cnn",
        projector_dim=256,
        action_output_activation="tanh",
        normalize_branch_embeddings=False,
        low_dim_dropout_prob=0.0,
    ):
        super(BCPolicy, self).__init__()
        self.vision_backbone = vision_backbone
        self.normalize_branch_embeddings = bool(normalize_branch_embeddings)
        self.low_dim_dropout_prob = float(low_dim_dropout_prob)
        if vision_backbone == "cnn":
            self.cnn_encoder = CNNEncoder(
                input_channels=image_channels,
                output_dim=projector_dim,
                image_size=image_size,
            )
        elif vision_backbone == "dinov3_vits16_frozen":
            self.cnn_encoder = FrozenDinoV3Encoder(
                input_channels=image_channels,
                output_dim=projector_dim,
            )
        elif vision_backbone == "depth_anything_small_frozen":
            self.cnn_encoder = FrozenDepthAnythingEncoder(
                input_channels=image_channels,
                output_dim=projector_dim,
            )
        elif vision_backbone == "dinov3_depth_anything_small_frozen":
            self.cnn_encoder = FrozenDinoDepthConcatEncoder(
                input_channels=image_channels,
                output_dim=projector_dim,
            )
        else:
            raise ValueError(
                "Unsupported vision_backbone: "
                f"{vision_backbone}. Expected one of {get_supported_vision_backbones()}."
            )
        if action_output_activation == "tanh":
            output_activation = nn.Tanh()
        elif action_output_activation == "identity":
            output_activation = nn.Identity()
        else:
            raise ValueError(
                "Unsupported action_output_activation: "
                f"{action_output_activation}. Expected one of ['tanh', 'identity']."
            )
        self.mlp_encoder = MLPEncoder(input_dim=low_dim_state_dim, output_dim=128)
        self.mlp_policy = nn.Sequential(
            nn.Linear(projector_dim + 128, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            output_activation,
        )

    def forward(self, image, low_dim_state):
        img_embedding = self.cnn_encoder(image)
        state_embedding = self.mlp_encoder(low_dim_state)
        if self.normalize_branch_embeddings:
            img_embedding = F.layer_norm(img_embedding, img_embedding.shape[-1:])
            state_embedding = F.layer_norm(state_embedding, state_embedding.shape[-1:])
        if self.training and self.low_dim_dropout_prob > 0.0:
            state_embedding = F.dropout(
                state_embedding,
                p=self.low_dim_dropout_prob,
                training=True,
            )
        fused = torch.cat([img_embedding, state_embedding], dim=-1)
        return self.mlp_policy(fused)
