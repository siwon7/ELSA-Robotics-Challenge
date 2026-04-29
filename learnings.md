# ELSA/FLAME Learnings

## Replay ceiling by action pipeline
- JV direct: close_box=1.0, slide=1.0, insert=0.0, scoop=0.2
- JP direct: close_box=1.0, slide=1.0, insert=0.8, scoop=0.9
- JP->JV servo (g20/c1.0/s2): close_box=0.9, slide=1.0, insert=0.9, scoop=0.9

## Current best same-env results (env0, 50ep)
- slide: DINO+Depth LoRA8 + diffusion + JV direct = SR 0.75
- close_box: DINO LoRA4 + diffusion + JV direct = SR 0.20
- insert: all configs = SR 0.00 (JV ceiling is 0%)
- scoop: JP keyframe4 + diffusion = SR 0.10

## Key code paths
- Agent/model: elsa_learning_agent/agent.py
- Action execution: elsa_learning_agent/utils.py (execute_action_with_adapter)
- Live rollout: elsa_learning_agent/live_rollout.py
- Same-env training: scripts/train_same_env_bcpolicy_probe.py
- Dataset loader: elsa_learning_agent/dataset/dataset_loader.py
- Experiment configs: experiments/*.yaml

## Template config for reference
- JV direct template: experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml
- JP direct template: experiments/action_pipeline_joint_position_direct_template.yaml

## Dataset paths
- Training: /mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets/training
- Eval: /mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets/eval
- Test: /mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets/test
- Results output: /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/
- Checkpoint output: /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints/

## Robot joint position limits (for JP configs)
- action_min: [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0]
- action_max: [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0]
- Last dim is gripper: 0.0 (closed) to 1.0 (open)

## Phase 1 - close+slide action+servo sweep (planned 2026-04-30)
- Planned same-env SR target for `close_box` and `slide_block_to_target`: 0.70-0.80 before Phase 2 vision/proprio ablations.
- Configs covered: `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jvdirect_splitgripper_e100.yaml`, `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect_splitgripper.yaml`, `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_splitgripper.yaml`, `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpdirect_chunk4exec2_splitgripper.yaml`, `experiments/close_box_sameenv_dino_depth_diffusion_lora8_jpservo_chunk4exec2_splitgripper.yaml`, `experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml`, `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpdirect.yaml`, and `experiments/slide_block_to_target_sameenv_dino_depth_diffusion_lora8_jpservo.yaml`.
- GPU plan: 1 GPU = 1 long run, 4 in parallel via `scripts/start_long_pair_sweep_tmux.sh`.
- Insert and scoop remain at SR 0.0/0.2 even with JP configs and require diagnostic work in a later batch.
