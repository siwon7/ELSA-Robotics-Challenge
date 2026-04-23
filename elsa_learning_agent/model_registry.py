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
    "volumedp_lite_dinov3_vits16": VisionBackboneSpec(
        name="volumedp_lite_dinov3_vits16",
        display_name="VolumeDP-lite with DINOv3 ViT-S/16",
        dependency="timm",
        upstream_source="timm pretrained vit_small_patch16_dinov3 + local volumetric token module",
        trainable_parts="DINO LoRA/projectors + volumetric token module + policy head",
        notes="Practical VolumeDP-style path using camera intrinsics/extrinsics and projected voxel tokens.",
    ),
    "volumedp_lite_camerafree_dinov3_vits16": VisionBackboneSpec(
        name="volumedp_lite_camerafree_dinov3_vits16",
        display_name="Camera-free VolumeDP-lite with DINOv3 ViT-S/16",
        dependency="timm",
        upstream_source="timm pretrained vit_small_patch16_dinov3 + local spatial token module",
        trainable_parts="DINO LoRA/projectors + spatial token module + policy head",
        notes="Benchmark-friendly variant without camera calibration. Uses DINO patch tokens and a proprio token in the decoder.",
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
