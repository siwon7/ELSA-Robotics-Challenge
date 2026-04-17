import inspect
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from elsa_learning_agent.kinematics import (
    EE_FEATURE_DIM,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    LOW_DIM_STATE_DIM,
    NUM_ARM_JOINTS,
)


DEFAULT_POLICY_NAME = "fk_camera_object"
DEFAULT_BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"
DEFAULT_BACKBONE_IMAGE_SIZE = 224
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16.0
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_NUM_OBJECT_TOKENS = 4
DEFAULT_NUM_LORA_BLOCKS = 4
DEFAULT_NUM_ATTENTION_HEADS = 8
DEFAULT_IMAGE_FEATURE_DIM = 384
DEFAULT_FRCR_LATENT_DIM = 256
DEFAULT_FRCR_PHASE_BINS = 4
DEFAULT_FRCR_RELATION_DIM = 4
DEFAULT_FRCR_RETRIEVAL_TOPK = 8
DEFAULT_FRCR_PERTURB_SCALE = 0.05
DEFAULT_FRCR_LOCAL_HIDDEN_DIM = 256
DEFAULT_FRCR_SEPARATE_GRIPPER_HEAD = False


def _expected_low_dim_state_dim():
    return NUM_ARM_JOINTS + EE_FEATURE_DIM + 1


def _require_fk_low_dim_state(low_dim_state_dim, policy_name):
    expected_dim = _expected_low_dim_state_dim()
    if low_dim_state_dim != expected_dim:
        raise ValueError(
            f"{policy_name} expects low_dim_state_dim={expected_dim}, got {low_dim_state_dim}"
        )


def build_policy_kwargs_from_config(config=None):
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        return {"policy_name": DEFAULT_POLICY_NAME}

    return {
        "policy_name": model_cfg.get("policy_name", DEFAULT_POLICY_NAME),
        "backbone_name": model_cfg.get("backbone_name", DEFAULT_BACKBONE_NAME),
        "backbone_image_size": int(
            model_cfg.get("backbone_image_size", DEFAULT_BACKBONE_IMAGE_SIZE)
        ),
        "image_feature_dim": int(
            model_cfg.get("image_feature_dim", DEFAULT_IMAGE_FEATURE_DIM)
        ),
        "lora_rank": int(model_cfg.get("lora_rank", DEFAULT_LORA_RANK)),
        "lora_alpha": float(model_cfg.get("lora_alpha", DEFAULT_LORA_ALPHA)),
        "lora_dropout": float(model_cfg.get("lora_dropout", DEFAULT_LORA_DROPOUT)),
        "num_object_tokens": int(
            model_cfg.get("num_object_tokens", DEFAULT_NUM_OBJECT_TOKENS)
        ),
        "num_lora_blocks": int(
            model_cfg.get("num_lora_blocks", DEFAULT_NUM_LORA_BLOCKS)
        ),
        "num_attention_heads": int(
            model_cfg.get("num_attention_heads", DEFAULT_NUM_ATTENTION_HEADS)
        ),
        "latent_dim": int(model_cfg.get("latent_dim", DEFAULT_FRCR_LATENT_DIM)),
        "num_phase_bins": int(
            model_cfg.get("num_phase_bins", DEFAULT_FRCR_PHASE_BINS)
        ),
        "relation_dim": int(
            model_cfg.get("relation_dim", DEFAULT_FRCR_RELATION_DIM)
        ),
        "retrieval_topk": int(
            model_cfg.get("retrieval_topk", DEFAULT_FRCR_RETRIEVAL_TOPK)
        ),
        "perturb_scale": float(
            model_cfg.get("perturb_scale", DEFAULT_FRCR_PERTURB_SCALE)
        ),
        "local_hidden_dim": int(
            model_cfg.get("local_hidden_dim", DEFAULT_FRCR_LOCAL_HIDDEN_DIM)
        ),
        "separate_gripper_head": bool(
            model_cfg.get(
                "separate_gripper_head", DEFAULT_FRCR_SEPARATE_GRIPPER_HEAD
            )
        ),
    }


def build_agent_kwargs(
    *,
    image_channels=3,
    low_dim_state_dim=LOW_DIM_STATE_DIM,
    action_dim=8,
    image_size=(128, 128),
    config=None,
):
    kwargs = {
        "image_channels": image_channels,
        "low_dim_state_dim": low_dim_state_dim,
        "action_dim": action_dim,
        "image_size": image_size,
    }
    kwargs.update(build_policy_kwargs_from_config(config))
    return kwargs


class Agent:
    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
        policy_name=DEFAULT_POLICY_NAME,
        **policy_kwargs,
    ):
        self.policy_name = policy_name
        policy_cls = get_policy_class(policy_name)
        accepted_kwargs = filter_policy_kwargs(policy_cls, policy_kwargs)
        self.policy = policy_cls(
            image_channels=image_channels,
            low_dim_state_dim=low_dim_state_dim,
            action_dim=action_dim,
            image_size=image_size,
            **accepted_kwargs,
        )

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def get_action(self, image, low_dim_state, return_aux=False, **forward_kwargs):
        kwargs = filter_forward_kwargs(self.policy, forward_kwargs)
        return self.policy(image, low_dim_state, return_aux=return_aux, **kwargs)

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

    def trainable_state_keys(self):
        trainable_names = {
            name for name, param in self.policy.named_parameters() if param.requires_grad
        }
        return [
            key for key in self.policy.state_dict().keys() if key in trainable_names
        ]

    def get_trainable_state_dict(self):
        state_dict = self.policy.state_dict()
        return OrderedDict(
            (key, state_dict[key]) for key in self.trainable_state_keys()
        )

    def load_trainable_state_dict(self, state_dict):
        current_state = self.policy.state_dict()
        current_state.update(state_dict)
        self.policy.load_state_dict(current_state, strict=False)
        return self

    def federated_state_keys(self):
        if hasattr(self.policy, "federated_state_keys"):
            return list(self.policy.federated_state_keys())
        return list(self.policy.state_dict().keys())

    def get_federated_state_dict(self):
        state_dict = self.policy.state_dict()
        return OrderedDict((key, state_dict[key]) for key in self.federated_state_keys())

    def load_federated_state_dict(self, state_dict):
        current_state = self.policy.state_dict()
        current_state.update(state_dict)
        self.policy.load_state_dict(current_state, strict=False)
        return self

    def local_state_keys(self):
        federated_keys = set(self.federated_state_keys())
        return [
            key for key in self.policy.state_dict().keys() if key not in federated_keys
        ]

    def get_local_state_dict(self):
        state_dict = self.policy.state_dict()
        return OrderedDict((key, state_dict[key]) for key in self.local_state_keys())

    def load_local_state_dict(self, state_dict):
        current_state = self.policy.state_dict()
        current_state.update(state_dict)
        self.policy.load_state_dict(current_state, strict=False)
        return self


def filter_policy_kwargs(policy_cls, policy_kwargs):
    signature = inspect.signature(policy_cls.__init__)
    accepted = set(signature.parameters.keys())
    accepted.discard("self")
    return {key: value for key, value in policy_kwargs.items() if key in accepted}


def filter_forward_kwargs(policy, forward_kwargs):
    signature = inspect.signature(policy.forward)
    accepted = set(signature.parameters.keys())
    accepted.discard("self")
    return {key: value for key, value in forward_kwargs.items() if key in accepted}


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
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


class FKBCPolicy(nn.Module):
    """Legacy FK baseline kept for checkpoint compatibility."""

    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
    ):
        super().__init__()
        _require_fk_low_dim_state(low_dim_state_dim, self.__class__.__name__)

        self.cnn_encoder = CNNEncoder(
            input_channels=image_channels,
            output_dim=256,
            image_size=image_size,
        )
        self.joint_encoder = MLPEncoder(
            input_dim=NUM_ARM_JOINTS, hidden_dim=128, output_dim=128
        )
        self.ee_pos_encoder = MLPEncoder(
            input_dim=EE_POS_DIM, hidden_dim=64, output_dim=64
        )
        self.ee_rot_encoder = MLPEncoder(
            input_dim=EE_ROT6D_DIM, hidden_dim=128, output_dim=64
        )
        self.gripper_encoder = MLPEncoder(input_dim=1, hidden_dim=32, output_dim=32)
        self.mlp_policy = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 64 + 32, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
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

    def forward(self, image, low_dim_state, return_aux=False):
        joint_state, ee_pos_state, ee_rot_state, gripper_state = (
            self._split_low_dim_state(low_dim_state)
        )

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
        action = self.mlp_policy(fused)
        if return_aux:
            return action, {}
        return action


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base layer.")

        self.base = base_layer
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)

        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        lora_update = self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return self.base(x) + lora_update


def _require_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "The foundation policy requires the `timm` package. "
            "Install project dependencies again so the DINOv2 backbone is available."
        ) from exc
    return timm


def apply_lora_to_vit(backbone, rank, alpha, dropout, num_lora_blocks):
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected a ViT-style backbone with `blocks`.")

    for param in backbone.parameters():
        param.requires_grad = False

    if num_lora_blocks <= 0:
        return

    blocks = list(backbone.blocks)
    start_idx = max(0, len(blocks) - num_lora_blocks)
    for block in blocks[start_idx:]:
        if hasattr(block, "attn"):
            for name in ("qkv", "proj"):
                layer = getattr(block.attn, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(
                        block.attn,
                        name,
                        LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout),
                    )


class FrozenBackboneCLSExtractor(nn.Module):
    """Extract frozen backbone CLS features for cached head-only training."""

    def __init__(
        self,
        backbone_name=DEFAULT_BACKBONE_NAME,
        backbone_image_size=DEFAULT_BACKBONE_IMAGE_SIZE,
    ):
        super().__init__()
        timm = _require_timm()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=backbone_image_size,
            dynamic_img_size=True,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone_image_size = backbone_image_size
        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            embed_dim = getattr(self.backbone, "embed_dim")
        self.embed_dim = embed_dim

        self.register_buffer(
            "foundation_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "foundation_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()

    def _prepare_foundation_input(self, image):
        image = ((image * 0.5) + 0.5).clamp(0.0, 1.0)
        if image.shape[-2:] != (
            self.backbone_image_size,
            self.backbone_image_size,
        ):
            image = F.interpolate(
                image,
                size=(self.backbone_image_size, self.backbone_image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (image - self.foundation_mean) / self.foundation_std

    def _split_vit_tokens(self, features):
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got shape {features.shape}")

        num_tokens = features.shape[1]
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return features.mean(dim=1), features

        side = int(math.isqrt(num_tokens - 1))
        if side * side == num_tokens - 1:
            return features[:, 0], features[:, 1:]

        raise ValueError(f"Could not infer patch tokens from shape {features.shape}")

    @torch.no_grad()
    def forward(self, image):
        foundation_input = self._prepare_foundation_input(image)
        features = self.backbone.forward_features(foundation_input)

        if isinstance(features, dict):
            patch_tokens = features.get("x_norm_patchtokens")
            cls_token = features.get("x_norm_clstoken")
            if patch_tokens is None:
                tokens = features.get("x_prenorm")
                if tokens is None:
                    raise ValueError(
                        "Unsupported backbone output dict. Expected DINOv2-style tokens."
                    )
                cls_token, _ = self._split_vit_tokens(tokens)
            elif cls_token is None:
                cls_token = patch_tokens.mean(dim=1)
            return cls_token

        if torch.is_tensor(features):
            if features.ndim == 3:
                cls_token, _ = self._split_vit_tokens(features)
                return cls_token
            if features.ndim == 4:
                patch_tokens = features.flatten(2).transpose(1, 2)
                return patch_tokens.mean(dim=1)

        raise ValueError(
            f"Unsupported backbone feature output type: {type(features).__name__}"
        )


class PatchObjectTokenizer(nn.Module):
    def __init__(self, embed_dim, num_tokens=4, token_dim=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.value_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.GELU(),
        )
        self.attn_proj = nn.Linear(embed_dim, num_tokens)
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim + 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )

    def _build_patch_coords(self, num_patches, device, dtype):
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                f"Expected a square patch grid, got {num_patches} patches."
            )

        y_coords = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(1, num_patches, 2)
        return coords

    def forward(self, patch_tokens, coord_shift=None):
        attn_logits = self.attn_proj(patch_tokens)
        attn = torch.softmax(attn_logits.transpose(1, 2), dim=-1)

        values = self.value_proj(patch_tokens)
        tokens = torch.bmm(attn, values)

        coords = self._build_patch_coords(
            num_patches=patch_tokens.shape[1],
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        ).expand(patch_tokens.shape[0], -1, -1)
        if coord_shift is not None:
            coords = coords + coord_shift.unsqueeze(1)
        pooled_coords = torch.bmm(attn, coords)

        object_tokens = self.token_proj(torch.cat([tokens, pooled_coords], dim=-1))
        entropy = -(attn.clamp_min(1e-6) * attn.clamp_min(1e-6).log()).sum(-1).mean()
        return object_tokens.flatten(1), entropy


class QueryTokenExtractor(nn.Module):
    def __init__(self, embed_dim, num_object_tokens, num_heads=8, output_dim=128):
        super().__init__()
        self.num_object_tokens = num_object_tokens
        self.output_dim = output_dim
        self.object_queries = nn.Parameter(torch.randn(num_object_tokens, embed_dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, patch_tokens, robot_query, object_query_bias):
        batch_size = patch_tokens.shape[0]
        object_queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        object_queries = object_queries + object_query_bias
        queries = torch.cat([robot_query.unsqueeze(1), object_queries], dim=1)

        attn_out, attn_weights = self.query_attn(
            self.query_norm(queries),
            self.patch_norm(patch_tokens),
            self.patch_norm(patch_tokens),
            need_weights=True,
            average_attn_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        query_tokens = self.out_proj(queries)
        return query_tokens, attn_weights.mean(dim=1)


class FKCameraObjectPolicy(nn.Module):
    """Object-centric FK policy using a frozen DINOv2-style backbone and LoRA."""

    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
        backbone_name=DEFAULT_BACKBONE_NAME,
        backbone_image_size=DEFAULT_BACKBONE_IMAGE_SIZE,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        num_object_tokens=DEFAULT_NUM_OBJECT_TOKENS,
        num_lora_blocks=DEFAULT_NUM_LORA_BLOCKS,
        num_attention_heads=DEFAULT_NUM_ATTENTION_HEADS,
    ):
        super().__init__()
        _require_fk_low_dim_state(low_dim_state_dim, self.__class__.__name__)

        self.image_channels = image_channels
        self.image_size = image_size
        self.backbone_image_size = backbone_image_size
        self.num_object_tokens = num_object_tokens

        timm = _require_timm()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=backbone_image_size,
            dynamic_img_size=True,
        )
        apply_lora_to_vit(
            backbone=self.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_lora_blocks=num_lora_blocks,
        )

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            embed_dim = getattr(self.backbone, "embed_dim")
        self.embed_dim = embed_dim

        self.register_buffer(
            "foundation_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "foundation_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.global_visual_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
        )

        self.joint_encoder = MLPEncoder(
            input_dim=NUM_ARM_JOINTS, hidden_dim=128, output_dim=128
        )
        self.ee_pos_encoder = MLPEncoder(
            input_dim=EE_POS_DIM, hidden_dim=64, output_dim=64
        )
        self.ee_rot_encoder = MLPEncoder(
            input_dim=EE_ROT6D_DIM, hidden_dim=128, output_dim=64
        )
        self.gripper_encoder = MLPEncoder(input_dim=1, hidden_dim=32, output_dim=32)

        state_feature_dim = 128 + 64 + 64 + 32
        self.view_encoder = nn.Sequential(
            nn.Linear(embed_dim + state_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.feature_adapter = nn.Linear(128, embed_dim * 2)
        self.camera_shift_head = nn.Linear(128, 2)
        self.coord_proj = nn.Linear(2, embed_dim)
        self.robot_query_proj = nn.Linear(128 + state_feature_dim, embed_dim)
        self.object_query_proj = nn.Linear(
            128,
            num_object_tokens * embed_dim,
        )
        self.query_token_extractor = QueryTokenExtractor(
            embed_dim=embed_dim,
            num_object_tokens=num_object_tokens,
            num_heads=num_attention_heads,
            output_dim=128,
        )
        self.robot_fk_proj = nn.Sequential(
            nn.Linear(state_feature_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.slot_temperature = nn.Parameter(torch.tensor(1.0))

        policy_input_dim = 256 + (num_object_tokens * 128) + state_feature_dim + 128
        self.mlp_policy = nn.Sequential(
            nn.Linear(policy_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
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

    def _prepare_foundation_input(self, image):
        image = ((image * 0.5) + 0.5).clamp(0.0, 1.0)
        if image.shape[-2:] != (
            self.backbone_image_size,
            self.backbone_image_size,
        ):
            image = F.interpolate(
                image,
                size=(self.backbone_image_size, self.backbone_image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (image - self.foundation_mean) / self.foundation_std

    def _split_vit_tokens(self, features):
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got shape {features.shape}")

        num_tokens = features.shape[1]
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return features.mean(dim=1), features

        side = int(math.isqrt(num_tokens - 1))
        if side * side == num_tokens - 1:
            return features[:, 0], features[:, 1:]

        raise ValueError(f"Could not infer patch tokens from shape {features.shape}")

    def _extract_backbone_tokens(self, image):
        features = self.backbone.forward_features(image)

        if isinstance(features, dict):
            patch_tokens = features.get("x_norm_patchtokens")
            cls_token = features.get("x_norm_clstoken")
            if patch_tokens is None:
                tokens = features.get("x_prenorm")
                if tokens is None:
                    raise ValueError(
                        "Unsupported backbone output dict. Expected DINOv2-style tokens."
                    )
                cls_token, patch_tokens = self._split_vit_tokens(tokens)
            elif cls_token is None:
                cls_token = patch_tokens.mean(dim=1)
            return cls_token, patch_tokens

        if torch.is_tensor(features):
            if features.ndim == 3:
                return self._split_vit_tokens(features)
            if features.ndim == 4:
                patch_tokens = features.flatten(2).transpose(1, 2)
                cls_token = patch_tokens.mean(dim=1)
                return cls_token, patch_tokens

        raise ValueError(
            f"Unsupported backbone feature output type: {type(features).__name__}"
        )

    def forward(self, image, low_dim_state, return_aux=False):
        joint_state, ee_pos_state, ee_rot_state, gripper_state = (
            self._split_low_dim_state(low_dim_state)
        )
        joint_embedding = self.joint_encoder(joint_state)
        ee_pos_embedding = self.ee_pos_encoder(ee_pos_state)
        ee_rot_embedding = self.ee_rot_encoder(ee_rot_state)
        gripper_embedding = self.gripper_encoder(gripper_state)
        state_embedding = torch.cat(
            [
                joint_embedding,
                ee_pos_embedding,
                ee_rot_embedding,
                gripper_embedding,
            ],
            dim=-1,
        )

        foundation_input = self._prepare_foundation_input(image)
        cls_token, patch_tokens = self._extract_backbone_tokens(foundation_input)

        view_code = self.view_encoder(torch.cat([cls_token, state_embedding], dim=-1))
        gamma, beta = self.feature_adapter(view_code).chunk(2, dim=-1)
        patch_tokens = (
            patch_tokens * (1.0 + 0.1 * gamma.unsqueeze(1))
            + 0.1 * beta.unsqueeze(1)
        )
        cls_token = cls_token * (1.0 + 0.1 * gamma) + 0.1 * beta

        camera_shift = 0.05 * torch.tanh(self.camera_shift_head(view_code))
        num_patches = patch_tokens.shape[1]
        side = int(math.isqrt(num_patches))
        if side * side == num_patches:
            coords = torch.linspace(
                -1.0,
                1.0,
                side,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
            coord_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, num_patches, 2)
            patch_tokens = patch_tokens + 0.05 * self.coord_proj(
                coord_grid.expand(patch_tokens.shape[0], -1, -1)
                + camera_shift.unsqueeze(1)
            )

        robot_query = self.robot_query_proj(torch.cat([view_code, state_embedding], dim=-1))
        object_query_bias = self.object_query_proj(view_code).view(
            patch_tokens.shape[0],
            self.num_object_tokens,
            self.embed_dim,
        )
        query_tokens, slot_attn = self.query_token_extractor(
            patch_tokens=patch_tokens,
            robot_query=robot_query,
            object_query_bias=object_query_bias,
        )
        robot_token = query_tokens[:, 0]
        object_tokens = query_tokens[:, 1:]
        attn_logits = slot_attn[:, 1:] / self.slot_temperature.clamp_min(0.1)
        attn = torch.softmax(attn_logits, dim=-1)
        attn_entropy = -(attn.clamp_min(1e-6) * attn.clamp_min(1e-6).log()).sum(-1).mean()
        global_visual = self.global_visual_proj(cls_token)
        object_token_flat = object_tokens.flatten(1)

        fused = torch.cat(
            [
                global_visual,
                object_token_flat,
                state_embedding,
                view_code,
            ],
            dim=-1,
        )
        action = self.mlp_policy(fused)
        canonical_tokens = torch.cat(
            [
                global_visual,
                robot_token,
                object_tokens.mean(dim=1),
            ],
            dim=-1,
        )
        fk_alignment = self.robot_fk_proj(state_embedding)
        slot_similarity = torch.matmul(
            F.normalize(object_tokens, dim=-1),
            F.normalize(object_tokens, dim=-1).transpose(1, 2),
        )
        eye = torch.eye(
            self.num_object_tokens,
            device=slot_similarity.device,
            dtype=slot_similarity.dtype,
        ).unsqueeze(0)
        slot_diversity = ((slot_similarity - eye) * (1.0 - eye)).pow(2).mean()
        modulation_reg = gamma.pow(2).mean() + beta.pow(2).mean()

        if return_aux:
            return action, {
                "camera_reg": camera_shift.pow(2).mean(),
                "attn_entropy": attn_entropy,
                "canonical_tokens": canonical_tokens,
                "robot_token": robot_token,
                "fk_alignment": fk_alignment,
                "object_tokens": object_tokens,
                "slot_diversity": slot_diversity,
                "modulation_reg": modulation_reg,
            }
        return action


class FKDinoCachedHeadPolicy(nn.Module):
    """Head-only policy operating on cached frozen DINO CLS features."""

    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
        image_feature_dim=DEFAULT_IMAGE_FEATURE_DIM,
        backbone_name=DEFAULT_BACKBONE_NAME,
        backbone_image_size=DEFAULT_BACKBONE_IMAGE_SIZE,
    ):
        super().__init__()
        del image_channels, image_size, backbone_name, backbone_image_size
        _require_fk_low_dim_state(low_dim_state_dim, self.__class__.__name__)

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.joint_encoder = MLPEncoder(
            input_dim=NUM_ARM_JOINTS, hidden_dim=128, output_dim=128
        )
        self.ee_pos_encoder = MLPEncoder(
            input_dim=EE_POS_DIM, hidden_dim=64, output_dim=64
        )
        self.ee_rot_encoder = MLPEncoder(
            input_dim=EE_ROT6D_DIM, hidden_dim=128, output_dim=64
        )
        self.gripper_encoder = MLPEncoder(input_dim=1, hidden_dim=32, output_dim=32)
        self.mlp_policy = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 64 + 32, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
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

    def forward(self, image, low_dim_state, return_aux=False):
        joint_state, ee_pos_state, ee_rot_state, gripper_state = (
            self._split_low_dim_state(low_dim_state)
        )
        visual_feature = self.visual_proj(image)
        joint_embedding = self.joint_encoder(joint_state)
        ee_pos_embedding = self.ee_pos_encoder(ee_pos_state)
        ee_rot_embedding = self.ee_rot_encoder(ee_rot_state)
        gripper_embedding = self.gripper_encoder(gripper_state)

        fused = torch.cat(
            [
                visual_feature,
                joint_embedding,
                ee_pos_embedding,
                ee_rot_embedding,
                gripper_embedding,
            ],
            dim=-1,
        )
        action = self.mlp_policy(fused)
        if return_aux:
            return action, {}
        return action


class FRCRPolicy(nn.Module):
    """Federated retrieval-conditioned recovery policy with local adapters."""

    LOCAL_STATE_PREFIXES = (
        "local_adapter.",
        "local_recovery.",
        "local_residual_head.",
        "local_arm_residual_head.",
        "local_gripper_residual_head.",
    )

    def __init__(
        self,
        image_channels=3,
        low_dim_state_dim=LOW_DIM_STATE_DIM,
        action_dim=4,
        image_size=(64, 64),
        latent_dim=DEFAULT_FRCR_LATENT_DIM,
        num_phase_bins=DEFAULT_FRCR_PHASE_BINS,
        relation_dim=DEFAULT_FRCR_RELATION_DIM,
        retrieval_topk=DEFAULT_FRCR_RETRIEVAL_TOPK,
        perturb_scale=DEFAULT_FRCR_PERTURB_SCALE,
        local_hidden_dim=DEFAULT_FRCR_LOCAL_HIDDEN_DIM,
        separate_gripper_head=DEFAULT_FRCR_SEPARATE_GRIPPER_HEAD,
    ):
        super().__init__()
        _require_fk_low_dim_state(low_dim_state_dim, self.__class__.__name__)

        self.latent_dim = latent_dim
        self.num_phase_bins = num_phase_bins
        self.relation_dim = relation_dim
        self.retrieval_topk = retrieval_topk
        self.perturb_scale = perturb_scale
        self.separate_gripper_head = separate_gripper_head

        self.visual_encoder = CNNEncoder(
            input_channels=image_channels,
            output_dim=latent_dim,
            image_size=image_size,
        )
        self.joint_encoder = MLPEncoder(
            input_dim=NUM_ARM_JOINTS, hidden_dim=128, output_dim=128
        )
        self.ee_pos_encoder = MLPEncoder(
            input_dim=EE_POS_DIM, hidden_dim=64, output_dim=64
        )
        self.ee_rot_encoder = MLPEncoder(
            input_dim=EE_ROT6D_DIM, hidden_dim=128, output_dim=64
        )
        self.gripper_encoder = MLPEncoder(input_dim=1, hidden_dim=32, output_dim=32)

        self.state_feature_dim = 128 + 64 + 64 + 32
        self.phase_embed_dim = latent_dim // 4
        self.relation_embed_dim = latent_dim // 4

        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim + self.state_feature_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_phase_bins),
        )
        self.phase_embedding = nn.Embedding(num_phase_bins, self.phase_embed_dim)
        self.relation_trunk = nn.Sequential(
            nn.Linear(latent_dim + self.state_feature_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.relation_head = nn.Linear(latent_dim, relation_dim)
        self.relation_proj = nn.Sequential(
            nn.Linear(relation_dim, self.relation_embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.relation_embed_dim),
        )

        retrieval_input_dim = latent_dim + self.phase_embed_dim + self.relation_embed_dim
        self.shared_query_proj = nn.Sequential(
            nn.Linear(retrieval_input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.shared_key_proj = nn.Sequential(
            nn.Linear(retrieval_input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.shared_value_proj = nn.Sequential(
            nn.Linear(retrieval_input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        self.local_adapter = nn.Sequential(
            nn.Linear(latent_dim, local_hidden_dim),
            nn.GELU(),
            nn.Linear(local_hidden_dim, latent_dim),
        )
        self.local_recovery = nn.Sequential(
            nn.Linear(latent_dim * 2, local_hidden_dim),
            nn.GELU(),
            nn.Linear(local_hidden_dim, latent_dim),
        )

        if self.separate_gripper_head:
            arm_action_dim = action_dim - 1
            if arm_action_dim <= 0:
                raise ValueError(
                    f"{self.__class__.__name__} requires action_dim >= 2 when "
                    "separate_gripper_head=True."
                )
            self.global_arm_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, arm_action_dim),
            )
            self.local_arm_residual_head = nn.Sequential(
                nn.Linear(latent_dim, local_hidden_dim),
                nn.GELU(),
                nn.Linear(local_hidden_dim, arm_action_dim),
            )
            self.global_gripper_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
            )
            self.local_gripper_residual_head = nn.Sequential(
                nn.Linear(latent_dim, local_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(local_hidden_dim // 2, 1),
            )
        else:
            self.global_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, action_dim),
            )
            self.local_residual_head = nn.Sequential(
                nn.Linear(latent_dim, local_hidden_dim),
                nn.GELU(),
                nn.Linear(local_hidden_dim, action_dim),
            )

    def federated_state_keys(self):
        return [
            key
            for key in self.state_dict().keys()
            if not key.startswith(self.LOCAL_STATE_PREFIXES)
        ]

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

    def _encode_shared_state(self, image, low_dim_state):
        joint_state, ee_pos_state, ee_rot_state, gripper_state = (
            self._split_low_dim_state(low_dim_state)
        )
        visual_embedding = self.visual_encoder(image)
        joint_embedding = self.joint_encoder(joint_state)
        ee_pos_embedding = self.ee_pos_encoder(ee_pos_state)
        ee_rot_embedding = self.ee_rot_encoder(ee_rot_state)
        gripper_embedding = self.gripper_encoder(gripper_state)
        state_embedding = torch.cat(
            [
                joint_embedding,
                ee_pos_embedding,
                ee_rot_embedding,
                gripper_embedding,
            ],
            dim=-1,
        )
        shared_latent = self.shared_encoder(
            torch.cat([visual_embedding, state_embedding], dim=-1)
        )
        return shared_latent, state_embedding, ee_pos_state, gripper_state

    def _phase_embedding_from_logits(self, phase_logits):
        phase_probs = torch.softmax(phase_logits, dim=-1)
        return phase_probs @ self.phase_embedding.weight

    def _build_relation_target(self, ee_pos_state, gripper_state, action_target):
        relation_target = torch.cat(
            [
                ee_pos_state,
                gripper_state,
                action_target[..., :3],
                action_target[..., -1:],
            ],
            dim=-1,
        )
        if relation_target.shape[-1] < self.relation_dim:
            relation_target = F.pad(
                relation_target,
                (0, self.relation_dim - relation_target.shape[-1]),
            )
        return relation_target[..., : self.relation_dim]

    def _phase_target_from_progress(self, progress):
        progress = progress.reshape(-1).clamp(0.0, 0.999999)
        phase_target = torch.floor(progress * self.num_phase_bins).long()
        return phase_target.clamp_(max=self.num_phase_bins - 1)

    def _retrieve_context(self, adapted_latent, phase_embedding, relation_embedding):
        batch_size = adapted_latent.shape[0]
        if batch_size <= 1:
            zeros = torch.zeros_like(adapted_latent)
            return zeros, adapted_latent, zeros

        retrieval_input = torch.cat(
            [adapted_latent, phase_embedding, relation_embedding], dim=-1
        )
        query = self.shared_query_proj(retrieval_input)
        key = self.shared_key_proj(retrieval_input)
        value = self.shared_value_proj(retrieval_input)

        z_dist = torch.cdist(query, key).pow(2)
        phase_dist = torch.cdist(phase_embedding, phase_embedding).pow(2)
        relation_dist = torch.cdist(relation_embedding, relation_embedding).pow(2)
        similarity = -(z_dist + 0.5 * phase_dist + 0.5 * relation_dist)
        similarity.fill_diagonal_(-1e9)

        topk = min(self.retrieval_topk, batch_size - 1)
        topk_scores, topk_indices = torch.topk(similarity, k=topk, dim=-1)

        gathered_values = torch.gather(
            value.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
        )
        gathered_latents = torch.gather(
            adapted_latent.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, adapted_latent.shape[-1]),
        )

        weights = torch.softmax(topk_scores / math.sqrt(query.shape[-1]), dim=-1)
        context = torch.sum(gathered_values * weights.unsqueeze(-1), dim=1)
        retrieved_latent = torch.sum(gathered_latents * weights.unsqueeze(-1), dim=1)
        return context, retrieved_latent, weights

    def forward(
        self,
        image,
        low_dim_state,
        return_aux=False,
        progress=None,
        action_target=None,
    ):
        shared_latent, state_embedding, ee_pos_state, gripper_state = (
            self._encode_shared_state(image, low_dim_state)
        )
        phase_logits = self.phase_head(shared_latent)
        phase_embedding = self._phase_embedding_from_logits(phase_logits)

        relation_features = self.relation_trunk(
            torch.cat([shared_latent, state_embedding], dim=-1)
        )
        relation_prediction = self.relation_head(relation_features)
        relation_embedding = self.relation_proj(relation_prediction)

        adapted_latent = shared_latent + self.local_adapter(shared_latent)
        context, retrieved_latent, retrieval_weights = self._retrieve_context(
            adapted_latent=adapted_latent,
            phase_embedding=phase_embedding,
            relation_embedding=relation_embedding,
        )
        corrected_latent = adapted_latent + self.local_recovery(
            torch.cat([adapted_latent, context], dim=-1)
        )

        if self.separate_gripper_head:
            nominal_arm_action = self.global_arm_head(corrected_latent)
            local_arm_residual = self.local_arm_residual_head(corrected_latent)
            arm_action = torch.tanh(nominal_arm_action + local_arm_residual)

            gripper_logit = self.global_gripper_head(
                corrected_latent
            ) + self.local_gripper_residual_head(corrected_latent)
            gripper_action = torch.sigmoid(gripper_logit)
            action = torch.cat([arm_action, gripper_action], dim=-1)
            local_residual_norm = (
                local_arm_residual.norm(dim=-1).mean()
                + self.local_gripper_residual_head(corrected_latent)
                .norm(dim=-1)
                .mean()
            )
        else:
            nominal_action = self.global_head(corrected_latent)
            local_residual = self.local_residual_head(corrected_latent)
            action = torch.tanh(nominal_action + local_residual)
            local_residual_norm = local_residual.norm(dim=-1).mean()

        if not return_aux:
            return action

        aux = {
            "phase_loss": corrected_latent.new_zeros(()),
            "relation_loss": corrected_latent.new_zeros(()),
            "retrieval_loss": F.mse_loss(
                corrected_latent, retrieved_latent.detach()
            ),
            "recovery_loss": corrected_latent.new_zeros(()),
            "smooth_loss": corrected_latent.new_zeros(()),
            "retrieval_mean_weight": retrieval_weights.mean()
            if retrieval_weights.numel() > 0
            else corrected_latent.new_zeros(()),
        }

        if progress is not None:
            phase_target = self._phase_target_from_progress(progress)
            aux["phase_loss"] = F.cross_entropy(phase_logits, phase_target)

        if action_target is not None:
            relation_target = self._build_relation_target(
                ee_pos_state=ee_pos_state,
                gripper_state=gripper_state,
                action_target=action_target,
            )
            aux["relation_loss"] = F.mse_loss(relation_prediction, relation_target)

        perturbed_latent = adapted_latent + (
            torch.randn_like(adapted_latent) * self.perturb_scale
        )
        recovered_latent = perturbed_latent + self.local_recovery(
            torch.cat([perturbed_latent, context.detach()], dim=-1)
        )
        aux["recovery_loss"] = F.mse_loss(recovered_latent, retrieved_latent.detach())

        base_distance = torch.norm(adapted_latent - retrieved_latent.detach(), dim=-1)
        corrected_distance = torch.norm(
            corrected_latent - retrieved_latent.detach(), dim=-1
        )
        aux["smooth_loss"] = F.relu(corrected_distance - 0.9 * base_distance).mean()
        aux["shared_latent_norm"] = shared_latent.norm(dim=-1).mean()
        aux["local_residual_norm"] = local_residual_norm
        return action, aux


class FRCRBCGripperPolicy(FRCRPolicy):
    """BC-first FRCR variant with a separate gripper head."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("separate_gripper_head", True)
        super().__init__(*args, **kwargs)


POLICY_REGISTRY = {
    "fkbc": FKBCPolicy,
    "fk_camera_object": FKCameraObjectPolicy,
    "fk_dino_cached_head": FKDinoCachedHeadPolicy,
    "frcr": FRCRPolicy,
    "frcr_bc_gripper": FRCRBCGripperPolicy,
}

POLICY_NAME_BY_CLASS = {
    policy_cls.__name__: policy_name
    for policy_name, policy_cls in POLICY_REGISTRY.items()
}


def get_policy_class(policy_name):
    try:
        return POLICY_REGISTRY[policy_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(
            f"Unsupported policy_name={policy_name!r}. Expected one of: {valid_names}"
        ) from exc


def get_policy_class_name(policy_name):
    return get_policy_class(policy_name).__name__


def infer_policy_name_from_model_path(model_path, default=DEFAULT_POLICY_NAME):
    file_name = os.path.basename(model_path)
    for class_name, policy_name in POLICY_NAME_BY_CLASS.items():
        if class_name in file_name:
            return policy_name
    return default


def policy_uses_cached_visual_features(policy_name):
    return policy_name == "fk_dino_cached_head"
