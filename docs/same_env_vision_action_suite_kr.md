# Same-Env Vision + Action Suite

이 문서는 FL에 들어가기 전에 `same-env`에서 `vision encoder`와 `action formulation`을 분리해서 확인하는 표준 실험 구성을 정리한다.

핵심 원칙:
- 먼저 `same-env`에서 policy가 실제로 학습되는지 확인
- 그 다음에만 FL generalization으로 확장
- action interface는 최종 목표에 맞춰 `joint_position -> benchmark joint_velocity servo`로 통일

## Stage 1: Action Sweep

목적:
- vision을 고정하고 어떤 action formulation이 `same-env` closed-loop에서 가장 잘 되는지 확인

고정:
- vision = `dinov3_vits16_frozen`
- execution = `joint_position_to_benchmark_joint_velocity_servo`

비교 대상:
- `onestep`
- `chunk3 + execute2`
- `chunk4 + execute2`
- `keyframe4`

관련 YAML:
- [onestep](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_onestep_dinov3_servo.yaml)
- [chunk3](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_chunk3_dinov3_servo.yaml)
- [chunk4](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_chunk4_dinov3_servo.yaml)
- [keyframe4](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_keyframe4_dinov3_servo.yaml)

추가 비교군:
- `action_sweep_dinov3_jpdirect`
  - 같은 4개 action formulation을 `joint_position_direct`로 다시 확인
- `JV direct` baseline
  - [onestep jv direct](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_onestep_dinov3_jvdirect.yaml)
  - `joint_velocity_direct` one-step이 same-env에서 어느 정도 버티는지 확인

## Stage 2: Vision Sweep

목적:
- Stage 1에서 고른 action 위에 어떤 visual encoder가 가장 잘 붙는지 확인

고정:
- action = `chunk4 + execute2`
- execution = `joint_position_to_benchmark_joint_velocity_servo`

비교 대상:
- `cnn`
- `dinov3_vits16_frozen`
- `depth_anything_small_frozen`
- `dinov3_depth_anything_small_frozen`

관련 YAML:
- [cnn](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_cnn_servo.yaml)
- [dinov3](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_dinov3_servo.yaml)
- [depth](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_depth_servo.yaml)
- [dino+depth](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_dino_depth_servo.yaml)

## Launcher

manifest:
- [slide_block_to_target_sameenv_suite.json](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_suite.json)

단일 실험 실행:
- [run_same_env_config_one_task.sh](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_same_env_config_one_task.sh)

tmux 그룹 실행:
- [start_same_env_suite_tmux.sh](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/start_same_env_suite_tmux.sh)

예시:
```bash
bash scripts/start_same_env_suite_tmux.sh action_sweep_dinov3 same_env_action_sweep 50 0
bash scripts/start_same_env_suite_tmux.sh action_sweep_dinov3_jpdirect same_env_action_sweep_jpdirect 50 0
bash scripts/start_same_env_suite_tmux.sh vision_sweep_chunk4exec2_servo same_env_vision_sweep 50 0
bash scripts/run_same_env_config_one_task.sh slide_block_to_target 0 \
  /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_onestep_dinov3_jvdirect.yaml \
  50 0 slide_sameenv_action_onestep_dinov3_jvdirect
```

## 결과 저장 위치

기본 artifact root:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts`

출력:
- results: `.../results/same_env_suite`
- checkpoints: `.../model_checkpoints/same_env_suite`
- logs: `.../logs/same_env_suite`

## 추천 순서

1. `action_sweep_dinov3`
2. `action_sweep_dinov3_jpdirect`
3. `JV direct` one-step baseline과 비교
4. same-env best action 선택
5. 필요하면 `vision_sweep_chunk4exec2_servo`
6. same-env에서 성능이 확인되면 FL로 이동
