# VolumeDP Paper Fidelity Checklist

Source of truth:
- Paper: `VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning` (`arXiv:2603.17720`)
- Main references used: Fig. 2, Sec. III, implementation details in Sec. IV, and Table IV ablations

Scope:
- This checklist compares the current repo against the paper's method design.
- "Paper-close" here means "closer to the paper while staying runnable in this codebase and hardware budget".
- It does not claim exact reproduction.

## Summary

The repo already matched the paper at a high level on:
- world-frame voxelized representation
- task-relevant spatial token generation
- multi-token diffusion decoding
- optional auxiliary supervision on voxel weights

The largest gaps before the paper-close pass were:
- no temporal two-frame input
- no decoder-side goal/text token
- weaker auxiliary supervision than the paper
- reduced grid/token/diffusion settings compared with the paper
- volumetric lifting implemented by feature sampling + depth fusion rather than the paper's deformable cross-attention

## Gap Table

| Item | Paper target | Repo before paper-close pass | Gap | Paper-close action |
|---|---|---|---|---|
| Visual input | two consecutive RGB frames `(t, t-1)` | single RGB frame | high | implemented temporal RGB pair support |
| Language conditioning for token generation | text token used in spatial token generation | learned goal token used only in weight head | medium | kept learned goal token, now also append its projected form to decoder as a proxy text token |
| Proprio conditioning | separate proprio token into decoder | token/global FiLM variants; default token path already existed | low | keep `token` mode for paper-close config |
| Volumetric lifting | Volume-Image deformable cross-attention | projected voxel sampling of DINO patch features + predicted depth fusion | high | not replaced; documented as a structural difference |
| Depth modality | paper is RGB-only | repo uses RGB + predicted depth inside encoder | high | kept as-is; closest existing stable path in this repo |
| Spatial token generation | goal-aware weights over voxels, softmax normalized | already present | low | retain and increase token count |
| Auxiliary supervision | BCE around EE position and gripper state change regions | BCE around EE position only | medium | keep EE-mask BCE; gripper-change region supervision deferred |
| Decoder conditioning | spatial tokens + text token + proprio token | spatial tokens + optional proprio token; no decoder goal token | medium | append projected goal token to decoder token set |
| Denoising steps | 100 | 20 | medium | increase to 100 in paper-close config |
| Voxel grid | 40x40x40 | 16x16x16 | medium | increase to 24x24x24 for runnable approximation |
| Spatial token count | 200 | 96 | medium | increase to 200 |
| Decoder token dim | paper projects tokens to 512 | repo typically uses 192 | medium | left unchanged for stability and memory |
| Multi-view cameras | paper can use multiple views depending benchmark | repo currently uses front camera context in this path | low | unchanged for this run |

## What Was Implemented

### 1. Temporal RGB pair support

Implemented:
- dataset path can emit `current frame || previous frame` stacked on channel dim
- live rollout path keeps the previous observation and feeds the same 2-frame structure at evaluation time
- utilities and checkpoint loaders now derive image channels from config/sample instead of assuming 3

Reason:
- This is the most meaningful paper-alignment change that was missing and is still low-risk enough to run in the current codebase.

### 2. Decoder-side proxy goal token

Implemented:
- if `volumedp_goal_token_dim > 0` and `volumedp_append_goal_token_to_decoder=true`, the encoder projects the learned goal token and appends it to `spatial_tokens`

Reason:
- Fig. 2 and Sec. III-C explicitly condition the decoder on spatial tokens, a proprio token, and a text token.
- This repo does not have a real text encoder path for same-task RLBench training.
- A learned goal token is the closest low-risk proxy available without introducing a new language stack.

### 3. Paper-close config changes

For the new config we moved toward the paper on:
- temporal input: enabled
- token count: increased
- voxel grid: increased
- denoising steps: increased
- EE auxiliary supervision: enabled
- proprio mode: explicit token mode

## Deferred Items

These were intentionally not implemented in the paper-close pass:

1. True deformable Volume-Image cross-attention
- This would require replacing the core volumetric lifting path, not just tuning config.

2. Real text encoder / task instruction token
- The current same-task training setup has no language data pipeline.
- A proper implementation should feed actual task instructions into both token generation and decoder.

3. Gripper-change region auxiliary supervision
- The paper mentions supervision around EE position and gripper state changes.
- The repo now supports EE-position supervision only.
- A faithful version should add an explicit interaction-region target based on gripper transitions.

4. Exact paper scale: 40x40x40 voxels with 512-d token projection
- Possible, but materially riskier for memory/runtime in this environment.
- The paper-close config uses a conservative approximation to keep the run practical.

## Figure 2-Aligned Proposal

Target experiment:
- `experiments/slide_block_to_target_sameenv_volumedp_full_dinov3_depth_lora8_jvdirect_paperclose_v1.yaml`

Design choices for this run:

| Figure 2 block | Paper intent | Paper-close implementation in this repo |
|---|---|---|
| Input observations | RGB frames at `t` and `t-1` plus proprio and task instruction | 2-frame RGB pair enabled, proprio token enabled, learned goal token used as proxy instruction token |
| Volumetric representation | image features lifted into a voxel volume | existing DINO + predicted-depth voxel lifting retained as the closest stable path |
| Spatial token generation | goal-aware voxel weighting, then top task-relevant tokens | keep voxel weighting head, increase token count to 200, enable EE auxiliary BCE |
| Multi-token decoder | spatial tokens + text token + proprio token, diffusion decoding | spatial tokens + appended projected goal token + proprio token, 100-step diffusion |

Practical config choices:
- `dataset.temporal_rgb_pair: true`
- `model.diffusion_num_steps: 100`
- `model.volumedp_grid_shape: [24, 24, 24]`
- `model.volumedp_num_spatial_tokens: 200`
- `model.proprio_visual_fusion_mode: token`
- `model.volumedp_append_goal_token_to_decoder: true`
- `model.ee_aux_loss_weight: 1.0`

Why this is the right "minimal paper-close" step:
- It fixes the two most important missing conditioning paths from the paper: temporal input and decoder-side goal/text-like token.
- It restores the paper's higher diffusion horizon and spatial token budget.
- It avoids replacing the core voxel lifting block, which is the highest-risk change and would delay experiments materially.

Why it is still not a reproduction:
- text conditioning is proxied, not real language
- volumetric lifting is not deformable cross-attention
- voxel scale and token projection dim are reduced for runtime safety
- auxiliary supervision still omits the gripper-transition region target

## Recommended Interpretation

Use the new experiment as:
- `VolumeDP paper-close approximation (temporal + 200 tokens + higher denoising + EE aux)`

Do not describe it as:
- `exact VolumeDP reproduction`

## Next Fidelity Step

If the paper-close run is promising, the next highest-value upgrade is:
1. replace the current voxel sampling lift with an explicit deformable cross-attention block
2. add a real task text token path
3. add gripper-transition-aware auxiliary supervision
