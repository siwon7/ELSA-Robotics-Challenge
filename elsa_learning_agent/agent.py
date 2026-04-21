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

import math

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
        use_adaln_head=False,
        adaln_hidden_dim=256,
        adaln_conditioning_mode="hybrid",
        use_dino_lora=False,
        dino_lora_rank=8,
        dino_lora_alpha=16.0,
        dino_lora_dropout=0.0,
        dino_lora_num_blocks=0,
        dino_lora_target_modules="qkv,proj",
        policy_head_type="mlp",
        diffusion_num_steps=20,
        diffusion_hidden_dim=512,
        diffusion_timestep_dim=128,
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
        use_adaln_head=use_adaln_head,
        adaln_hidden_dim=adaln_hidden_dim,
        adaln_conditioning_mode=adaln_conditioning_mode,
        use_dino_lora=use_dino_lora,
        dino_lora_rank=dino_lora_rank,
        dino_lora_alpha=dino_lora_alpha,
        dino_lora_dropout=dino_lora_dropout,
        dino_lora_num_blocks=dino_lora_num_blocks,
        dino_lora_target_modules=dino_lora_target_modules,
        policy_head_type=policy_head_type,
        diffusion_num_steps=diffusion_num_steps,
        diffusion_hidden_dim=diffusion_hidden_dim,
        diffusion_timestep_dim=diffusion_timestep_dim,
        )

    def train(self, ):
        self.policy.train()

    def eval(self, ):
        self.policy.eval()

    def get_action(self, image, low_dim_state):
        # Get the action from the policy
        return self.policy(image, low_dim_state) # Assuming the policy is a function

    def compute_loss(self, image, low_dim_state, action, criterion=None):
        return self.policy.compute_loss(image, low_dim_state, action, criterion=criterion)
    
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


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.base_layer(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_update = F.linear(lora_hidden, self.lora_B) * self.scale
        return base + lora_update


def _parse_lora_target_modules(raw_value) -> set[str]:
    if isinstance(raw_value, str):
        tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in raw_value if str(token).strip()]
    target_modules = set(tokens or ["qkv", "proj"])
    invalid = sorted(target_modules - {"qkv", "proj"})
    if invalid:
        raise ValueError(
            f"Unsupported dino_lora_target_modules={invalid}. Expected any of ['qkv', 'proj']."
        )
    return target_modules


def _inject_dino_lora(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    num_blocks: int,
    target_modules,
):
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("DINO backbone does not expose transformer blocks for LoRA injection.")
    total_blocks = len(blocks)
    if total_blocks <= 0:
        return
    num_blocks = int(max(0, min(total_blocks, num_blocks)))
    if num_blocks == 0:
        return
    target_modules = _parse_lora_target_modules(target_modules)
    for block_idx in range(total_blocks - num_blocks, total_blocks):
        attn = blocks[block_idx].attn
        if "qkv" in target_modules and isinstance(attn.qkv, nn.Linear):
            attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        if "proj" in target_modules and isinstance(attn.proj, nn.Linear):
            attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha, dropout=dropout)


class FrozenDinoV3Encoder(nn.Module):
    """Frozen DINOv3-S/16 image encoder with a small trainable projector."""

    def __init__(
        self,
        input_channels=3,
        output_dim=256,
        use_dino_lora=False,
        dino_lora_rank=8,
        dino_lora_alpha=16.0,
        dino_lora_dropout=0.0,
        dino_lora_num_blocks=0,
        dino_lora_target_modules="qkv,proj",
    ):
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
        self.use_dino_lora = bool(use_dino_lora)
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
        if use_dino_lora:
            _inject_dino_lora(
                self.backbone,
                rank=int(dino_lora_rank),
                alpha=float(dino_lora_alpha),
                dropout=float(dino_lora_dropout),
                num_blocks=int(dino_lora_num_blocks),
                target_modules=dino_lora_target_modules,
            )
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        x = ((x * 0.5) + 0.5).clamp(0.0, 1.0)
        x = (x - self.backbone_mean) / self.backbone_std
        if self.use_dino_lora:
            features = self.backbone(x)
        else:
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

    def __init__(
        self,
        input_channels=3,
        output_dim=256,
        use_dino_lora=False,
        dino_lora_rank=8,
        dino_lora_alpha=16.0,
        dino_lora_dropout=0.0,
        dino_lora_num_blocks=0,
        dino_lora_target_modules="qkv,proj",
    ):
        super(FrozenDinoDepthConcatEncoder, self).__init__()
        dino_dim = output_dim // 2
        depth_dim = output_dim - dino_dim
        self.dino_encoder = FrozenDinoV3Encoder(
            input_channels=input_channels,
            output_dim=dino_dim,
            use_dino_lora=use_dino_lora,
            dino_lora_rank=dino_lora_rank,
            dino_lora_alpha=dino_lora_alpha,
            dino_lora_dropout=dino_lora_dropout,
            dino_lora_num_blocks=dino_lora_num_blocks,
            dino_lora_target_modules=dino_lora_target_modules,
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


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    if dim <= 0:
        raise ValueError(f"timestep embedding dim must be positive, got {dim}.")
    half = dim // 2
    device = timesteps.device
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiffusionActionHead(nn.Module):
    """Small denoising MLP over action vectors conditioned on context and timestep."""

    def __init__(self, action_dim: int, context_dim: int, hidden_dim: int = 512, timestep_dim: int = 128):
        super().__init__()
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.timestep_dim = int(timestep_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"diffusion_hidden_dim must be positive, got {self.hidden_dim}.")
        if self.timestep_dim <= 0:
            raise ValueError(f"diffusion_timestep_dim must be positive, got {self.timestep_dim}.")

        input_dim = self.action_dim + self.context_dim + self.timestep_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, noisy_action: torch.Tensor, context: torch.Tensor, timesteps: torch.Tensor):
        timestep_embedding = _sinusoidal_timestep_embedding(timesteps, self.timestep_dim)
        hidden = torch.cat([noisy_action, context, timestep_embedding], dim=-1)
        return self.net(hidden)


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
        use_adaln_head=False,
        adaln_hidden_dim=256,
        adaln_conditioning_mode="hybrid",
        use_dino_lora=False,
        dino_lora_rank=8,
        dino_lora_alpha=16.0,
        dino_lora_dropout=0.0,
        dino_lora_num_blocks=0,
        dino_lora_target_modules="qkv,proj",
        policy_head_type="mlp",
        diffusion_num_steps=20,
        diffusion_hidden_dim=512,
        diffusion_timestep_dim=128,
    ):
        super(BCPolicy, self).__init__()
        self.vision_backbone = vision_backbone
        self.normalize_branch_embeddings = bool(normalize_branch_embeddings)
        self.low_dim_dropout_prob = float(low_dim_dropout_prob)
        self.use_adaln_head = bool(use_adaln_head)
        self.adaln_hidden_dim = int(adaln_hidden_dim)
        self.adaln_conditioning_mode = str(adaln_conditioning_mode)
        if self.adaln_hidden_dim <= 0:
            raise ValueError(
                f"adaln_hidden_dim must be positive, got {self.adaln_hidden_dim}."
            )
        if self.adaln_conditioning_mode not in {"proprio", "image", "hybrid"}:
            raise ValueError(
                "adaln_conditioning_mode must be one of "
                "['proprio', 'image', 'hybrid'], "
                f"got {self.adaln_conditioning_mode}."
            )
        self.policy_head_type = str(policy_head_type)
        if self.policy_head_type not in {"mlp", "diffusion"}:
            raise ValueError(
                "policy_head_type must be one of ['mlp', 'diffusion'], "
                f"got {self.policy_head_type}."
            )
        self.diffusion_num_steps = int(diffusion_num_steps)
        self.diffusion_hidden_dim = int(diffusion_hidden_dim)
        self.diffusion_timestep_dim = int(diffusion_timestep_dim)
        if self.policy_head_type == "diffusion" and self.diffusion_num_steps <= 1:
            raise ValueError(
                "diffusion_num_steps must be greater than 1 when using diffusion head."
            )
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
                use_dino_lora=use_dino_lora,
                dino_lora_rank=dino_lora_rank,
                dino_lora_alpha=dino_lora_alpha,
                dino_lora_dropout=dino_lora_dropout,
                dino_lora_num_blocks=dino_lora_num_blocks,
                dino_lora_target_modules=dino_lora_target_modules,
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
                use_dino_lora=use_dino_lora,
                dino_lora_rank=dino_lora_rank,
                dino_lora_alpha=dino_lora_alpha,
                dino_lora_dropout=dino_lora_dropout,
                dino_lora_num_blocks=dino_lora_num_blocks,
                dino_lora_target_modules=dino_lora_target_modules,
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
        self.state_feature_dim = 128
        self.mlp_encoder = MLPEncoder(
            input_dim=low_dim_state_dim,
            output_dim=self.state_feature_dim,
        )
        fused_dim = projector_dim + self.state_feature_dim
        self.policy_fc1 = nn.Linear(fused_dim, 512)
        self.policy_fc2 = None
        self.output_activation = output_activation
        if self.use_adaln_head:
            if self.adaln_conditioning_mode == "proprio":
                adaln_input_dim = self.state_feature_dim
            elif self.adaln_conditioning_mode == "image":
                adaln_input_dim = projector_dim
            else:
                adaln_input_dim = fused_dim
            self.adaln_context = nn.Sequential(
                nn.Linear(adaln_input_dim, self.adaln_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.adaln_hidden_dim, 1024),
            )
        else:
            self.adaln_context = None
        if self.policy_head_type == "mlp":
            self.policy_fc2 = nn.Linear(512, action_dim)
            self.diffusion_head = None
        else:
            self.diffusion_head = DiffusionActionHead(
                action_dim=action_dim,
                context_dim=512,
                hidden_dim=self.diffusion_hidden_dim,
                timestep_dim=self.diffusion_timestep_dim,
            )
            betas = torch.linspace(1e-4, 0.02, self.diffusion_num_steps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer("diffusion_betas", betas, persistent=False)
            self.register_buffer("diffusion_alphas", alphas, persistent=False)
            self.register_buffer("diffusion_alpha_cumprod", alpha_cumprod, persistent=False)
            self.register_buffer(
                "diffusion_sqrt_alpha_cumprod",
                torch.sqrt(alpha_cumprod),
                persistent=False,
            )
            self.register_buffer(
                "diffusion_sqrt_one_minus_alpha_cumprod",
                torch.sqrt(1.0 - alpha_cumprod),
                persistent=False,
            )

    def _encode_context(self, image, low_dim_state):
        img_embedding = self.cnn_encoder(image)
        state_embedding = self.mlp_encoder(low_dim_state)
        if self.normalize_branch_embeddings:
            img_embedding = F.layer_norm(img_embedding, img_embedding.shape[-1:])
            state_embedding = F.layer_norm(state_embedding, state_embedding.shape[-1:])
        clean_img_embedding = img_embedding
        clean_state_embedding = state_embedding
        if self.training and self.low_dim_dropout_prob > 0.0:
            state_embedding = F.dropout(
                state_embedding,
                p=self.low_dim_dropout_prob,
                training=True,
            )
        fused = torch.cat([img_embedding, state_embedding], dim=-1)
        if self.adaln_conditioning_mode == "proprio":
            adaln_condition = clean_state_embedding
        elif self.adaln_conditioning_mode == "image":
            adaln_condition = clean_img_embedding
        else:
            adaln_condition = torch.cat([clean_img_embedding, clean_state_embedding], dim=-1)
        hidden = self.policy_fc1(fused)
        if self.adaln_context is not None:
            gamma, beta = self.adaln_context(adaln_condition).chunk(2, dim=-1)
            hidden = F.layer_norm(hidden, hidden.shape[-1:])
            hidden = hidden * (1.0 + gamma) + beta
        return F.relu(hidden)

    def _forward_mlp(self, context):
        hidden = self.policy_fc2(context)
        return self.output_activation(hidden)

    def _diffusion_predict_noise(self, noisy_action, context, timesteps):
        return self.diffusion_head(noisy_action, context, timesteps)

    def _sample_diffusion(self, context):
        batch_size = context.shape[0]
        action_dim = self.diffusion_head.action_dim
        current = torch.randn(batch_size, action_dim, device=context.device, dtype=context.dtype)
        for step in reversed(range(self.diffusion_num_steps)):
            t = torch.full((batch_size,), step, device=context.device, dtype=torch.long)
            predicted_noise = self._diffusion_predict_noise(current, context, t)
            alpha_t = self.diffusion_alphas[step]
            alpha_cumprod_t = self.diffusion_alpha_cumprod[step]
            beta_t = self.diffusion_betas[step]
            current = (
                current - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
            ) / torch.sqrt(alpha_t)
        return current.clamp(-1.0, 1.0)

    def compute_loss(self, image, low_dim_state, target_action, criterion=None):
        context = self._encode_context(image, low_dim_state)
        if self.policy_head_type == "mlp":
            predicted = self._forward_mlp(context)
            if criterion is None:
                return F.mse_loss(predicted, target_action)
            return criterion(predicted, target_action)

        batch_size = target_action.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.diffusion_num_steps,
            size=(batch_size,),
            device=target_action.device,
        )
        noise = torch.randn_like(target_action)
        sqrt_alpha_cumprod_t = self.diffusion_sqrt_alpha_cumprod[timesteps].unsqueeze(-1)
        sqrt_one_minus_t = self.diffusion_sqrt_one_minus_alpha_cumprod[timesteps].unsqueeze(-1)
        noisy_action = sqrt_alpha_cumprod_t * target_action + sqrt_one_minus_t * noise
        predicted_noise = self._diffusion_predict_noise(noisy_action, context, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def forward(self, image, low_dim_state):
        context = self._encode_context(image, low_dim_state)
        if self.policy_head_type == "mlp":
            return self._forward_mlp(context)
        return self._sample_diffusion(context)
