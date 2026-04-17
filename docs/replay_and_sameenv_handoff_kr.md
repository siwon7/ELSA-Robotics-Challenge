# Replay + Same-Env Handoff

이 문서는 다른 사람이 현재 코드와 실험 구성을 그대로 이해하고, 같은 세팅으로 자신의 실험을 시작할 수 있도록 정리한 handoff 문서다.

리포지토리:
- upstream: https://github.com/KTH-RPL/ELSA-Robotics-Challenge
- fork: https://github.com/siwon7/ELSA-Robotics-Challenge

기준:
- 이 문서를 포함한 commit을 checkout해서 쓰는 것을 기준으로 한다.
- 데이터셋 경로와 artifact 경로는 로컬 환경에 맞게 조정해야 한다.

## 1. 현재 핵심 결론

지금까지 확인한 결론은 두 가지다.

1. action injection 자체는 어느 정도 정리됐다.
- 저장된 성공 trajectory를 replay하면 action pipeline에 따라 높은 성공률이 나온다.
- 즉 actuator/interface 자체가 주병목은 아니다.

2. learned policy는 아직 action을 충분히 잘 예측하지 못한다.
- 특히 `slide_block_to_target`은 replay ceiling은 높지만, 학습된 policy는 same-env와 FL eval에서 아직 약하다.
- 그래서 FL에 바로 확장하기 전에 `same-env`에서 `vision + action` 조합을 먼저 검증하는 구조로 바꿨다.

## 2. Action Pipeline 3개

관련 코드:
- [utils.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/utils.py)
- [dataset_loader.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/dataset/dataset_loader.py)
- [action_pipeline_presets_kr.md](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/action_pipeline_presets_kr.md)

현재 실험에서 쓰는 action pipeline은 세 가지다.

### A. `joint_velocity_direct`

의미:
- 학습 target: `obs.joint_velocities + next_obs.gripper_open`
- 실행: RLBench/Colosseum 공식 `JointVelocity + Discrete gripper`

언제 쓰나:
- 공식 baseline과 같은 interface를 유지하고 싶을 때

### B. `joint_position_direct`

의미:
- 학습 target: `next_obs.joint_positions + next_obs.gripper_open`
- 실행: `JointPosition(absolute) + Discrete gripper`

언제 쓰나:
- replay upper bound를 먼저 확인하고 싶을 때
- current dataset에서 가장 안정적인 공통 replay semantics를 쓰고 싶을 때

### C. `joint_position_to_benchmark_joint_velocity_servo`

의미:
- 학습 target: `next_obs.joint_positions + next_obs.gripper_open`
- 실행 interface: benchmark 쪽 `JointVelocity + Discrete gripper`
- 중간에 `joint_position -> joint_velocity servo adapter`를 거친다.

기본값:
- `joint_velocity_servo_gain = 20.0`
- `joint_velocity_servo_clip = 1.0`
- `joint_velocity_servo_steps = 2`
- `joint_velocity_servo_tolerance = 0.01`

변환:
```text
v = clip(gain * (q_target - q_current), -clip, clip)
```

언제 쓰나:
- 학습은 JP target으로 하고 싶지만, 최종 benchmark execution은 JV interface로 맞추고 싶을 때

## 3. Replay Upper Bound

현재까지 정리된 replay 성공률은 아래와 같다.

| pipeline | close_box | slide | insert | scoop |
|---|---:|---:|---:|---:|
| `joint_velocity_direct` | `1.0` | `1.0` | `0.0` | `0.2` |
| `joint_position_direct` | `1.0` | `1.0` | `0.8` | `0.9` |
| `joint_position_to_benchmark_joint_velocity_servo` `g20/c1.0/s2` | `0.9` | `1.0` | `0.9` | `0.9` |

해석:
- 공통 replay ceiling은 아직 `joint_position_direct`가 가장 안정적이다.
- benchmark JV strict 경로가 필요하면 `joint_position_to_benchmark_joint_velocity_servo`가 현재 가장 균형적이다.
- `slide`만 보면 세 방식 모두 replay는 된다.

중요한 해석:
- 저장된 성공 trajectory replay는 된다는 뜻이다.
- 즉 지금은 “행동을 넣는 법”보다 “정책이 그 행동을 얼마나 잘 예측하느냐”가 더 중요하다.

## 4. Same-Env Suite를 만든 이유

기존에는 FL full run에서 바로 성능을 봤다. 이 구조는 문제가 있었다.

- action formulation이 안 좋은지
- vision backbone이 약한지
- FL이 non-IID 때문에 망가지는지

이 세 가지가 한 번에 섞였다.

그래서 지금은 순서를 바꿨다.

1. same-env에서 action formulation 먼저 비교
2. same-env에서 vision encoder 비교
3. same-env에서 실제로 learning signal이 나오면
4. 그 다음에만 FL generalization으로 올림

관련 문서:
- [same_env_vision_action_suite_kr.md](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/same_env_vision_action_suite_kr.md)

## 5. Same-Env 실험 구성

manifest:
- [slide_block_to_target_sameenv_suite.json](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_suite.json)

### Stage 1. Action Sweep

vision을 고정하고 action만 비교한다.

고정:
- task: `slide_block_to_target`
- env: `env0`
- vision: `dinov3_vits16_frozen`
- execution: `joint_position_to_benchmark_joint_velocity_servo`

비교:
- one-step
- chunk3 + execute2
- chunk4 + execute2
- keyframe4

YAML:
- [onestep](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_onestep_dinov3_servo.yaml)
- [chunk3](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_chunk3_dinov3_servo.yaml)
- [chunk4](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_chunk4_dinov3_servo.yaml)
- [keyframe4](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_keyframe4_dinov3_servo.yaml)

### Stage 2. Vision Sweep

action을 고정하고 vision encoder만 비교한다.

고정:
- action: `chunk4 + execute2`
- execution: `joint_position_to_benchmark_joint_velocity_servo`

비교:
- `cnn`
- `dinov3_vits16_frozen`
- `depth_anything_small_frozen`
- `dinov3_depth_anything_small_frozen`

YAML:
- [cnn](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_cnn_servo.yaml)
- [dinov3](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_dinov3_servo.yaml)
- [depth](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_depth_servo.yaml)
- [dino+depth](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_vision_chunk4exec2_dino_depth_servo.yaml)

## 6. 실행 방법

단일 실험:
- [run_same_env_config_one_task.sh](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_same_env_config_one_task.sh)

tmux 그룹 실행:
- [start_same_env_suite_tmux.sh](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/start_same_env_suite_tmux.sh)

예시:
```bash
bash scripts/start_same_env_suite_tmux.sh action_sweep_dinov3 same_env_action_sweep 50 0
bash scripts/start_same_env_suite_tmux.sh vision_sweep_chunk4exec2_servo same_env_vision_sweep 50 0
```

이 명령은 각각 4개 실험을 tmux window로 띄운다.

환경 전제:
- `scripts/prepare_live_eval_env.sh`가 내부에서 호출된다.
- 기본 conda env는 `${ELSA_ENV_NAME:-elsa_challenge}`다.
- GPU는 window index 순서대로 `0,1,2,3`에 배정된다.

## 7. 결과가 저장되는 위치

artifact root 기본값:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts`

same-env suite 출력:
- results: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite`
- checkpoints: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints/same_env_suite`
- logs: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/same_env_suite`

probe 결과에는 다음이 같이 저장된다.
- `result.json`
- `resolved_config.yaml`
- `vision_backbone`
- `action_chunk_len`
- `action_keyframe_horizon`
- `receding_horizon_execute_steps`

즉 나중에 결과만 봐도 어떤 세팅으로 돌렸는지 역추적이 가능하다.

## 8. 새 사람이 따라할 때 권장 순서

1. repo checkout
2. dataset path와 artifact path 확인
3. `validate_experiment_config.py`로 대표 YAML validation
4. same-env action sweep
5. best action 선택
6. same-env vision sweep
7. same-env에서 성능이 나오면 그때 FL run

대표 validator 예시:
```bash
source scripts/common_env.sh
conda activate "${ELSA_ENV_NAME:-elsa_challenge}"
python scripts/validate_experiment_config.py \
  --config experiments/slide_block_to_target_sameenv_action_chunk4_dinov3_servo.yaml \
  --task slide_block_to_target \
  --env-id 0 \
  --split train \
  --normalize
```

## 9. CPU/GPU 사용 해석

same-env probe를 돌릴 때 GPU 사용률이 기대보다 낮아 보일 수 있다. 이건 이상 동작이 아니다.

현재 same-env probe의 특성:
- backbone이 frozen인 경우가 많다
- batch가 작다
- training보다 RLBench live eval과 simulator step이 크다
- gzip pickle decode와 Python dataloader 비용이 있다

즉 same-env probe는 생각보다 GPU-heavy가 아니라 CPU/simulator-heavy다.

실제 해석:
- GPU는 policy forward/backward 때만 잠깐 쓴다
- RLBench/CoppeliaSim과 live eval 구간은 주로 CPU를 쓴다
- 그래서 `nvidia-smi`상 GPU util이 낮아도 실험이 정상일 수 있다

정리:
- same-env local probe는 “GPU만 쓰는” 실험이 아니다
- 실제 병목은 자주 CPU + simulator 쪽이다

## 10. 현재 추천 baseline

지금 same-env 실험을 설계한 기준 baseline은 다음이다.

- action family: `chunk` 계열
- execution: `joint_position_to_benchmark_joint_velocity_servo`
- strong vision candidate: `dinov3_vits16_frozen`
- 비교 vision: `depth_anything_small_frozen`, `dinov3_depth_anything_small_frozen`

다만 이 문서는 추천값을 고정하는 문서가 아니라, 다른 사람이 같은 구조로 실험을 이어받게 만드는 문서다.

따라서 새 사람이 해야 할 일은:
- same-env action sweep 결과를 먼저 보고
- 그 위에 vision sweep을 얹고
- 마지막에 FL로 확장하는 것이다.
