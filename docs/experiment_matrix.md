# Phase 1 Experiment Matrix

This document is the single source of truth for the current same-env sweep plan. The immediate goal is to push `close_box` and `slide_block_to_target` to same-env SR 0.70-0.80 before moving on to vision/proprio ablation work, so Phase 1 focuses on the action-interface and servo choices most likely to raise those two tasks quickly.

## Replay ceilings

| Task | JV direct | JP direct | JP->JV servo |
| --- | --- | --- | --- |
| `close_box` | 1.0 | 1.0 | 0.9 |
| `slide_block_to_target` | 1.0 | 1.0 | 1.0 |
| `insert_onto_square_peg` | 0.0 | 0.8 | 0.9 |
| `scoop_with_spatula` | 0.2 | 0.9 | 0.9 |

## Phase 1 - Action sweep on close_box and slide

| Task | Action interface | Splitgripper | Chunk | Config path | Status | Best SR |
| --- | --- | --- | --- | --- | --- | --- |
| `close_box` | JV direct | yes | none | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jvdirect_splitgripper_e100.yaml` | done |  |
| `close_box` | JP direct | yes | none | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect_splitgripper.yaml` | pending |  |
| `close_box` | JP->JV servo | yes | none | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_splitgripper.yaml` | pending |  |
| `close_box` | JP direct | yes | 4/2 | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2_splitgripper.yaml` | pending |  |
| `close_box` | JP->JV servo | yes | 4/2 | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2_splitgripper.yaml` | pending |  |
| `slide_block_to_target` | JV direct | no | none | `experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml` | done |  |
| `slide_block_to_target` | JP direct | no | none | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml` | pending |  |
| `slide_block_to_target` | JP->JV servo | no | none | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo.yaml` | pending |  |
| `close_box` | JP->JV servo | yes | none | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_g30c10_splitgripper.yaml` | pending |  |
| `close_box` | JP->JV servo | yes | none | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_g20c05_splitgripper.yaml` | pending |  |
| `slide_block_to_target` | JP->JV servo | no | none | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_g30c10.yaml` | pending |  |
| `slide_block_to_target` | JP->JV servo | no | none | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_g20c05.yaml` | pending |  |
| `slide_block_to_target` | JP direct | no | 4/2 | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2.yaml` | pending |  |
| `slide_block_to_target` | JP->JV servo | no | 4/2 | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2.yaml` | pending |  |
| `phase1A` | Wave A status | mixed | mixed | `Wave A baseline close_box/slide sweep remains in flight` | pending |  |

## Wave B - Servo gain/clip sweep

| Task | Config path | Servo (gain/clip) | GPU | Status |
| --- | --- | --- | --- | --- |
| `close_box` | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_g30c10_splitgripper.yaml` | `30.0/1.0` | `0` | pending |
| `close_box` | `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_g20c05_splitgripper.yaml` | `20.0/0.5` | `1` | pending |
| `slide_block_to_target` | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_g30c10.yaml` | `30.0/1.0` | `2` | pending |
| `slide_block_to_target` | `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo_g20c05.yaml` | `20.0/0.5` | `3` | pending |

## Wave C - Slide vision ablation (Phase 2 candidate)

| Vision backbone | Config path | Status |
| --- | --- | --- |
| `cnn` | `experiments/slide_block_to_target_sameenv_cnn_diffusion_jvdirect.yaml` | pending |
| `dinov3_vits16_frozen` | `experiments/slide_block_to_target_sameenv_dinov3_frozen_diffusion_jvdirect.yaml` | pending |
| `dinov3_depth_anything_small_frozen` | `experiments/slide_block_to_target_sameenv_dino_depth_frozen_diffusion_jvdirect.yaml` | pending |

## Phase 2 - VolumeDP-FL with per-client extrinsics

Per-client camera extrinsics are realistic in federated robotics because each client knows its own calibration, and camera pose is already treated as a perturbation factor in FLAME (Table I, `Δx/Δy/Δz: (-0.05, 0.05)`). The existing `volumedp_lite` encoder will be extended into `VolumeDPFullDinoDepthEncoder` by Claude, not by ralph, to add Depth-Anything voxel lift, goal-aware softmax token weights in place of top-k selection, and an EE-mask BCE auxiliary supervision hook.

## Phase 2 - Vision and proprio ablation

Phase 2 will be designed after the Phase 1 winners are known so the ablations stay focused on the strongest action-interface candidates instead of spreading effort across losing sweep branches.

## Phase 3 - Federated baseline and strategy

Phase 3 depends on the Phase 2 outcome and will define the federated baseline and strategy only after the best close_box/slide recipe is stable.

## How to run

```bash
EPOCHS=100 EVAL_EPISODES=20 SEED=0 ENV_ID=0 \
scripts/start_long_pair_sweep_tmux.sh phase1_close_slide phase1_close_slide 0,1,2,3 \
  close_box:experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect_splitgripper.yaml \
  close_box:experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_splitgripper.yaml \
  slide_block_to_target:experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml \
  slide_block_to_target:experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo.yaml

python scripts/aggregate_sameenv_sweep_results.py \
  --tasks close_box slide_block_to_target \
  --out results/phase1_close_slide_summary.csv

scripts/start_wave_B_tmux.sh

python scripts/wave_summary.py --waves phase1A
```
