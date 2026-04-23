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
        volumedp_volume_bounds=None,
        volumedp_grid_shape=(8, 8, 8),
        volumedp_num_spatial_tokens=32,
        volumedp_decoder_layers=2,
        volumedp_decoder_heads=4,
        volumedp_action_token_dim=8,
        proprio_visual_fusion_mode="token",
        proprio_visual_fusion_hidden_dim=256,
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
        volumedp_volume_bounds=volumedp_volume_bounds,
        volumedp_grid_shape=volumedp_grid_shape,
        volumedp_num_spatial_tokens=volumedp_num_spatial_tokens,
        volumedp_decoder_layers=volumedp_decoder_layers,
        volumedp_decoder_heads=volumedp_decoder_heads,
        volumedp_action_token_dim=volumedp_action_token_dim,
        proprio_visual_fusion_mode=proprio_visual_fusion_mode,
        proprio_visual_fusion_hidden_dim=proprio_visual_fusion_hidden_dim,
        )

    def train(self, ):
        self.policy.train()

    def eval(self, ):
        self.policy.eval()

    def get_action(self, image, low_dim_state, obs_context=None):
        # Get the action from the policy
        return self.policy(image, low_dim_state, obs_context=obs_context) # Assuming the policy is a function

    def compute_loss(self, image, low_dim_state, action, criterion=None, obs_context=None):
        return self.policy.compute_loss(
            image,
            low_dim_state,
            action,
            criterion=criterion,
            obs_context=obs_context,
        )
    
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


def _extract_vit_tokens(backbone: nn.Module, image_batch: torch.Tensor):
    features = backbone.forward_features(image_batch)
    cls_token = None
    patch_tokens = None
    num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1) or 1)

    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            patch_tokens = features["x_norm_patchtokens"]
            cls_token = features.get("x_norm_clstoken")
        elif "x_prenorm" in features:
            sequence = features["x_prenorm"]
            cls_token = sequence[:, 0]
            patch_tokens = sequence[:, 1:]
        else:
            tensor_values = [value for value in features.values() if torch.is_tensor(value)]
            if not tensor_values:
                raise ValueError("Unsupported DINO forward_features output: no tensor values found.")
            sequence = tensor_values[0]
            if sequence.ndim == 3 and sequence.shape[1] > 1:
                cls_token = sequence[:, 0]
                patch_tokens = sequence[:, 1:]
            elif sequence.ndim == 3:
                patch_tokens = sequence
            else:
                raise ValueError(
                    f"Unsupported DINO forward_features tensor shape: {tuple(sequence.shape)}"
                )
    elif torch.is_tensor(features):
        if features.ndim == 3 and features.shape[1] > num_prefix_tokens:
            cls_token = features[:, 0]
            patch_tokens = features[:, num_prefix_tokens:]
        elif features.ndim == 3:
            patch_tokens = features
        elif features.ndim == 2:
            cls_token = features
            patch_tokens = features.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported DINO forward_features output shape: {tuple(features.shape)}")
    else:
        raise ValueError(f"Unsupported DINO forward_features output type: {type(features)!r}")

    if patch_tokens is None:
        raise ValueError("Failed to extract patch tokens from DINO backbone.")
    if cls_token is None:
        cls_token = patch_tokens.mean(dim=1)

    grid_size = getattr(getattr(backbone, "patch_embed", None), "grid_size", None)
    patch_count = int(patch_tokens.shape[1])
    if isinstance(grid_size, tuple) and int(grid_size[0]) * int(grid_size[1]) == patch_count:
        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    else:
        grid_h = int(math.sqrt(patch_count))
        grid_w = int(patch_count // max(1, grid_h))
        if grid_h * grid_w != patch_count:
            raise ValueError(
                f"Cannot infer patch grid for token count {patch_count}. "
                "Expected a perfect square or backbone.patch_embed.grid_size."
            )

    feature_map = patch_tokens.view(patch_tokens.shape[0], grid_h, grid_w, patch_tokens.shape[-1])
    feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
    return cls_token, patch_tokens, feature_map


def _make_voxel_centers(bounds, grid_shape):
    xmin, ymin, zmin, xmax, ymax, zmax = [float(value) for value in bounds]
    gx, gy, gz = [int(value) for value in grid_shape]
    xs = torch.linspace(xmin, xmax, steps=gx)
    ys = torch.linspace(ymin, ymax, steps=gy)
    zs = torch.linspace(zmin, zmax, steps=gz)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def _make_patch_centers(grid_h: int, grid_w: int):
    ys = torch.linspace(0.0, 1.0, steps=grid_h)
    xs = torch.linspace(0.0, 1.0, steps=grid_w)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)


def _project_world_points_to_normalized_grid(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
):
    if extrinsics.shape[-2:] == (4, 4):
        world_to_camera = torch.linalg.inv(extrinsics)
    elif extrinsics.shape[-2:] == (3, 4):
        eye_row = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=extrinsics.device,
            dtype=extrinsics.dtype,
        ).view(1, 1, 4)
        world_to_camera = torch.cat([extrinsics, eye_row.expand(extrinsics.shape[0], -1, -1)], dim=1)
    else:
        raise ValueError(
            f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}. Expected (B,4,4) or (B,3,4)."
        )

    batch_size, num_points = intrinsics.shape[0], points_world.shape[0]
    points_h = torch.cat(
        [
            points_world.unsqueeze(0).expand(batch_size, -1, -1),
            torch.ones(batch_size, num_points, 1, device=points_world.device, dtype=points_world.dtype),
        ],
        dim=-1,
    )
    camera_points = torch.einsum("bij,bnj->bni", world_to_camera, points_h)[..., :3]
    positive_depth = camera_points[..., 2] > 1e-4

    projected = torch.einsum("bij,bnj->bni", intrinsics, camera_points)
    u = projected[..., 0] / projected[..., 2].clamp_min(1e-6)
    v = projected[..., 1] / projected[..., 2].clamp_min(1e-6)

    grid_x = 2.0 * (u / max(1.0, float(image_width - 1))) - 1.0
    grid_y = 2.0 * (v / max(1.0, float(image_height - 1))) - 1.0
    valid = positive_depth & (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid, valid


class VolumeDPLiteDinoEncoder(nn.Module):
    """Practical VolumeDP-style encoder with projected voxel tokens from a DINO feature map."""

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
        volume_bounds=None,
        grid_shape=(8, 8, 8),
        num_spatial_tokens=32,
    ):
        super().__init__()
        if input_channels != 3:
            raise ValueError(
                "volumedp_lite_dinov3_vits16 expects 3-channel RGB input, "
                f"got input_channels={input_channels}."
            )
        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as exc:
            raise ImportError(
                "timm is required when vision_backbone='volumedp_lite_dinov3_vits16'."
            ) from exc

        self.backbone = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=True,
            num_classes=0,
        )
        self.use_dino_lora = bool(use_dino_lora)
        self.output_dim = int(output_dim)
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.global_projector = nn.Sequential(
            nn.Linear(self.backbone.num_features, self.output_dim),
            nn.ReLU(),
        )
        self.token_projector = nn.Linear(self.backbone.num_features, self.output_dim)
        self.voxel_pos_mlp = nn.Sequential(
            nn.Linear(3, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.token_scorer = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, 1),
        )

        bounds = volume_bounds or [-0.45, -0.55, 0.70, 0.45, 0.55, 1.35]
        self.register_buffer(
            "voxel_centers",
            _make_voxel_centers(bounds, grid_shape).float(),
            persistent=False,
        )
        self.register_buffer(
            "volume_bounds",
            torch.tensor(bounds, dtype=torch.float32),
            persistent=False,
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

    def forward(self, x, obs_context=None):
        if obs_context is None:
            raise ValueError("VolumeDP-lite encoder requires obs_context with camera intrinsics/extrinsics.")
        intrinsics = obs_context.get("front_camera_intrinsics")
        extrinsics = obs_context.get("front_camera_extrinsics")
        if intrinsics is None or extrinsics is None:
            raise ValueError(
                "VolumeDP-lite encoder requires front_camera_intrinsics and front_camera_extrinsics."
            )

        x = ((x * 0.5) + 0.5).clamp(0.0, 1.0)
        x = (x - self.backbone_mean) / self.backbone_std
        if self.use_dino_lora:
            cls_token, _, feature_map = _extract_vit_tokens(self.backbone, x)
        else:
            with torch.no_grad():
                cls_token, _, feature_map = _extract_vit_tokens(self.backbone, x)

        batch_size = x.shape[0]
        image_height = int(x.shape[-2])
        image_width = int(x.shape[-1])
        grid, valid = _project_world_points_to_normalized_grid(
            self.voxel_centers.to(x.device),
            intrinsics.to(x.device),
            extrinsics.to(x.device),
            image_height,
            image_width,
        )
        sampled = F.grid_sample(
            feature_map,
            grid.view(batch_size, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        voxel_tokens = self.token_projector(sampled)

        bounds_min = self.volume_bounds[:3].to(x.device)
        bounds_max = self.volume_bounds[3:].to(x.device)
        normalized_voxels = (self.voxel_centers.to(x.device) - bounds_min) / (bounds_max - bounds_min).clamp_min(1e-6)
        voxel_tokens = voxel_tokens + self.voxel_pos_mlp(normalized_voxels).unsqueeze(0)

        scores = self.token_scorer(voxel_tokens).squeeze(-1)
        scores = scores.masked_fill(~valid, float("-inf"))
        topk = min(self.num_spatial_tokens, voxel_tokens.shape[1])
        topk_indices = scores.topk(topk, dim=1).indices
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, voxel_tokens.shape[-1])
        spatial_tokens = torch.gather(voxel_tokens, dim=1, index=gather_index)
        global_embedding = self.global_projector(cls_token)
        return {
            "global_embedding": global_embedding,
            "spatial_tokens": spatial_tokens,
        }


class CameraFreeVolumeDPLiteDinoEncoder(nn.Module):
    """Camera-free VolumeDP-inspired encoder using DINO patch tokens as spatial tokens."""

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
        num_spatial_tokens=32,
    ):
        super().__init__()
        if input_channels != 3:
            raise ValueError(
                "volumedp_lite_camerafree_dinov3_vits16 expects 3-channel RGB input, "
                f"got input_channels={input_channels}."
            )
        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as exc:
            raise ImportError(
                "timm is required when vision_backbone='volumedp_lite_camerafree_dinov3_vits16'."
            ) from exc

        self.backbone = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=True,
            num_classes=0,
        )
        self.use_dino_lora = bool(use_dino_lora)
        self.output_dim = int(output_dim)
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.global_projector = nn.Sequential(
            nn.Linear(self.backbone.num_features, self.output_dim),
            nn.ReLU(),
        )
        self.token_projector = nn.Linear(self.backbone.num_features, self.output_dim)
        self.patch_pos_mlp = nn.Sequential(
            nn.Linear(2, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.token_scorer = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, 1),
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

    def forward(self, x, obs_context=None):
        del obs_context
        x = ((x * 0.5) + 0.5).clamp(0.0, 1.0)
        x = (x - self.backbone_mean) / self.backbone_std
        if self.use_dino_lora:
            cls_token, patch_tokens, feature_map = _extract_vit_tokens(self.backbone, x)
        else:
            with torch.no_grad():
                cls_token, patch_tokens, feature_map = _extract_vit_tokens(self.backbone, x)

        del feature_map
        batch_size = x.shape[0]
        token_features = self.token_projector(patch_tokens)
        patch_count = token_features.shape[1]
        grid_h = int(math.sqrt(patch_count))
        grid_w = int(patch_count // max(1, grid_h))
        if grid_h * grid_w != patch_count:
            raise ValueError(
                f"Camera-free VolumeDP-lite expects square patch layout, got patch_count={patch_count}."
            )
        patch_positions = _make_patch_centers(grid_h, grid_w).to(x.device)
        token_features = token_features + self.patch_pos_mlp(patch_positions).unsqueeze(0)

        scores = self.token_scorer(token_features).squeeze(-1)
        topk = min(self.num_spatial_tokens, token_features.shape[1])
        topk_indices = scores.topk(topk, dim=1).indices
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, token_features.shape[-1])
        spatial_tokens = torch.gather(token_features, dim=1, index=gather_index)
        global_embedding = self.global_projector(cls_token)
        return {
            "global_embedding": global_embedding,
            "spatial_tokens": spatial_tokens,
        }


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


class MultiTokenDiffusionActionHead(nn.Module):
    """Transformer decoder conditioned on spatial tokens, closer to VolumeDP than a single-vector denoiser."""

    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        token_context_dim: int,
        hidden_dim: int = 512,
        timestep_dim: int = 128,
        action_token_dim: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.token_context_dim = int(token_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.timestep_dim = int(timestep_dim)
        self.action_token_dim = int(max(1, action_token_dim))
        if self.action_dim % self.action_token_dim != 0:
            self.action_token_dim = self.action_dim
        self.num_action_tokens = max(1, self.action_dim // self.action_token_dim)

        self.action_proj = nn.Linear(self.action_token_dim, self.hidden_dim)
        self.context_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.token_context_proj = nn.Linear(self.token_context_dim, self.hidden_dim)
        self.timestep_proj = nn.Linear(self.timestep_dim, self.hidden_dim)
        self.action_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_action_tokens, self.hidden_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(decoder_heads),
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(decoder_layers),
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.action_token_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        context: torch.Tensor,
        spatial_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        batch_size = noisy_action.shape[0]
        timestep_embedding = _sinusoidal_timestep_embedding(timesteps, self.timestep_dim)
        action_tokens = noisy_action.view(batch_size, self.num_action_tokens, self.action_token_dim)
        action_tokens = self.action_proj(action_tokens)
        action_tokens = action_tokens + self.action_pos_embed[:, : self.num_action_tokens]
        action_tokens = action_tokens + self.context_proj(context).unsqueeze(1)
        action_tokens = action_tokens + self.timestep_proj(timestep_embedding).unsqueeze(1)

        memory = self.token_context_proj(spatial_tokens)
        decoded = self.decoder(tgt=action_tokens, memory=memory)
        denoised = self.output_proj(decoded)
        return denoised.reshape(batch_size, self.action_dim)


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
        volumedp_volume_bounds=None,
        volumedp_grid_shape=(8, 8, 8),
        volumedp_num_spatial_tokens=32,
        volumedp_decoder_layers=2,
        volumedp_decoder_heads=4,
        volumedp_action_token_dim=8,
        proprio_visual_fusion_mode="token",
        proprio_visual_fusion_hidden_dim=256,
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
        self.volumedp_action_token_dim = int(volumedp_action_token_dim)
        if self.policy_head_type not in {"mlp", "diffusion"}:
            raise ValueError(
                "policy_head_type must be one of ['mlp', 'diffusion'], "
                f"got {self.policy_head_type}."
            )
        self.diffusion_num_steps = int(diffusion_num_steps)
        self.diffusion_hidden_dim = int(diffusion_hidden_dim)
        self.diffusion_timestep_dim = int(diffusion_timestep_dim)
        self.proprio_visual_fusion_mode = str(proprio_visual_fusion_mode or "token")
        self.proprio_visual_fusion_hidden_dim = int(proprio_visual_fusion_hidden_dim)
        valid_proprio_visual_fusion_modes = {
            "none",
            "token",
            "global_film",
            "token_film",
            "global_token_film",
        }
        if self.proprio_visual_fusion_mode not in valid_proprio_visual_fusion_modes:
            raise ValueError(
                "proprio_visual_fusion_mode must be one of "
                f"{sorted(valid_proprio_visual_fusion_modes)}, "
                f"got {self.proprio_visual_fusion_mode}."
            )
        if self.proprio_visual_fusion_hidden_dim <= 0:
            raise ValueError(
                "proprio_visual_fusion_hidden_dim must be positive, "
                f"got {self.proprio_visual_fusion_hidden_dim}."
            )
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
        elif vision_backbone == "volumedp_lite_dinov3_vits16":
            self.cnn_encoder = VolumeDPLiteDinoEncoder(
                input_channels=image_channels,
                output_dim=projector_dim,
                use_dino_lora=use_dino_lora,
                dino_lora_rank=dino_lora_rank,
                dino_lora_alpha=dino_lora_alpha,
                dino_lora_dropout=dino_lora_dropout,
                dino_lora_num_blocks=dino_lora_num_blocks,
                dino_lora_target_modules=dino_lora_target_modules,
                volume_bounds=volumedp_volume_bounds,
                grid_shape=volumedp_grid_shape,
                num_spatial_tokens=volumedp_num_spatial_tokens,
            )
        elif vision_backbone == "volumedp_lite_camerafree_dinov3_vits16":
            self.cnn_encoder = CameraFreeVolumeDPLiteDinoEncoder(
                input_channels=image_channels,
                output_dim=projector_dim,
                use_dino_lora=use_dino_lora,
                dino_lora_rank=dino_lora_rank,
                dino_lora_alpha=dino_lora_alpha,
                dino_lora_dropout=dino_lora_dropout,
                dino_lora_num_blocks=dino_lora_num_blocks,
                dino_lora_target_modules=dino_lora_target_modules,
                num_spatial_tokens=volumedp_num_spatial_tokens,
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
        self.proprio_token_proj = None
        if vision_backbone.startswith("volumedp_lite") and self.proprio_visual_fusion_mode in {
            "token",
            "token_film",
            "global_token_film",
        }:
            self.proprio_token_proj = nn.Linear(self.state_feature_dim, projector_dim)
        self.proprio_global_film = None
        if self.proprio_visual_fusion_mode in {"global_film", "global_token_film"}:
            self.proprio_global_film = nn.Sequential(
                nn.Linear(self.state_feature_dim, self.proprio_visual_fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proprio_visual_fusion_hidden_dim, 2 * projector_dim),
            )
            nn.init.zeros_(self.proprio_global_film[-1].weight)
            nn.init.zeros_(self.proprio_global_film[-1].bias)
        self.proprio_token_film = None
        if self.proprio_visual_fusion_mode in {"token_film", "global_token_film"}:
            self.proprio_token_film = nn.Sequential(
                nn.Linear(self.state_feature_dim, self.proprio_visual_fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proprio_visual_fusion_hidden_dim, 2 * projector_dim),
            )
            nn.init.zeros_(self.proprio_token_film[-1].weight)
            nn.init.zeros_(self.proprio_token_film[-1].bias)
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
            self.multitoken_diffusion_head = None
        else:
            self.diffusion_head = DiffusionActionHead(
                action_dim=action_dim,
                context_dim=512,
                hidden_dim=self.diffusion_hidden_dim,
                timestep_dim=self.diffusion_timestep_dim,
            )
            if vision_backbone.startswith("volumedp_lite"):
                self.multitoken_diffusion_head = MultiTokenDiffusionActionHead(
                    action_dim=action_dim,
                    context_dim=512,
                    token_context_dim=projector_dim,
                    hidden_dim=self.diffusion_hidden_dim,
                    timestep_dim=self.diffusion_timestep_dim,
                    action_token_dim=self.volumedp_action_token_dim,
                    decoder_layers=int(volumedp_decoder_layers),
                    decoder_heads=int(volumedp_decoder_heads),
                )
            else:
                self.multitoken_diffusion_head = None
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

    def _encode_context(self, image, low_dim_state, obs_context=None):
        encoder_output = self.cnn_encoder(image, obs_context) if self.vision_backbone.startswith("volumedp_lite") else self.cnn_encoder(image)
        spatial_tokens = None
        if isinstance(encoder_output, dict):
            img_embedding = encoder_output["global_embedding"]
            spatial_tokens = encoder_output.get("spatial_tokens")
        else:
            img_embedding = encoder_output
        state_embedding = self.mlp_encoder(low_dim_state)
        if self.normalize_branch_embeddings:
            img_embedding = F.layer_norm(img_embedding, img_embedding.shape[-1:])
            state_embedding = F.layer_norm(state_embedding, state_embedding.shape[-1:])
            if spatial_tokens is not None:
                spatial_tokens = F.layer_norm(spatial_tokens, spatial_tokens.shape[-1:])
        if self.proprio_global_film is not None:
            gamma, beta = self.proprio_global_film(state_embedding).chunk(2, dim=-1)
            img_embedding = img_embedding * (1.0 + gamma) + beta
        if spatial_tokens is not None and self.proprio_token_film is not None:
            gamma, beta = self.proprio_token_film(state_embedding).chunk(2, dim=-1)
            spatial_tokens = spatial_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        if spatial_tokens is not None and self.proprio_token_proj is not None:
            proprio_token = self.proprio_token_proj(state_embedding).unsqueeze(1)
            spatial_tokens = torch.cat([spatial_tokens, proprio_token], dim=1)
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
        return F.relu(hidden), spatial_tokens

    def _forward_mlp(self, context):
        hidden = self.policy_fc2(context)
        return self.output_activation(hidden)

    def _diffusion_predict_noise(self, noisy_action, context, timesteps, spatial_tokens=None):
        if self.multitoken_diffusion_head is not None and spatial_tokens is not None:
            return self.multitoken_diffusion_head(noisy_action, context, spatial_tokens, timesteps)
        return self.diffusion_head(noisy_action, context, timesteps)

    def _sample_diffusion(self, context, spatial_tokens=None):
        batch_size = context.shape[0]
        action_dim = self.diffusion_head.action_dim
        current = torch.randn(batch_size, action_dim, device=context.device, dtype=context.dtype)
        for step in reversed(range(self.diffusion_num_steps)):
            t = torch.full((batch_size,), step, device=context.device, dtype=torch.long)
            predicted_noise = self._diffusion_predict_noise(current, context, t, spatial_tokens=spatial_tokens)
            alpha_t = self.diffusion_alphas[step]
            alpha_cumprod_t = self.diffusion_alpha_cumprod[step]
            beta_t = self.diffusion_betas[step]
            current = (
                current - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
            ) / torch.sqrt(alpha_t)
        return current.clamp(-1.0, 1.0)

    def compute_loss(self, image, low_dim_state, target_action, criterion=None, obs_context=None):
        context, spatial_tokens = self._encode_context(image, low_dim_state, obs_context=obs_context)
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
        predicted_noise = self._diffusion_predict_noise(
            noisy_action,
            context,
            timesteps,
            spatial_tokens=spatial_tokens,
        )
        return F.mse_loss(predicted_noise, noise)

    def forward(self, image, low_dim_state, obs_context=None):
        context, spatial_tokens = self._encode_context(
            image,
            low_dim_state,
            obs_context=obs_context,
        )
        if self.policy_head_type == "mlp":
            return self._forward_mlp(context)
        return self._sample_diffusion(context, spatial_tokens=spatial_tokens)
