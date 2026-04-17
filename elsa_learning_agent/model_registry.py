from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionBackboneSpec:
    name: str
    display_name: str
    dependency: str
    upstream_source: str
    trainable_parts: str
    notes: str


VISION_BACKBONE_REGISTRY: dict[str, VisionBackboneSpec] = {
    "cnn": VisionBackboneSpec(
        name="cnn",
        display_name="Local CNN baseline",
        dependency="torch",
        upstream_source="local implementation",
        trainable_parts="full encoder + policy head",
        notes="Fast baseline. Weakest option under strong color/view shifts.",
    ),
    "dinov3_vits16_frozen": VisionBackboneSpec(
        name="dinov3_vits16_frozen",
        display_name="Frozen DINOv3 ViT-S/16",
        dependency="timm",
        upstream_source="timm pretrained vit_small_patch16_dinov3",
        trainable_parts="projector + policy head only",
        notes="Recommended first global FL backbone under color/camera variation.",
    ),
    "depth_anything_small_frozen": VisionBackboneSpec(
        name="depth_anything_small_frozen",
        display_name="Frozen Depth Anything Small",
        dependency="transformers",
        upstream_source="LiheYoung/depth-anything-small-hf",
        trainable_parts="depth projector + policy head only",
        notes="Useful as a depth-centric branch. Usually second-stage after DINO.",
    ),
    "dinov3_depth_anything_small_frozen": VisionBackboneSpec(
        name="dinov3_depth_anything_small_frozen",
        display_name="Frozen DINOv3 + Depth Anything concat",
        dependency="timm + transformers",
        upstream_source="timm pretrained vit_small_patch16_dinov3 + LiheYoung/depth-anything-small-hf",
        trainable_parts="both projectors + policy head only",
        notes="Higher-capacity visual stack. Use after the DINO-only baseline is stable.",
    ),
}


def get_supported_vision_backbones() -> list[str]:
    return sorted(VISION_BACKBONE_REGISTRY.keys())


def get_vision_backbone_spec(name: str) -> VisionBackboneSpec:
    try:
        return VISION_BACKBONE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported vision_backbone: {name}. "
            f"Expected one of {get_supported_vision_backbones()}"
        ) from exc
