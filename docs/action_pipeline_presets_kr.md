# Action Pipeline Presets

이 문서는 ELSA/FLAME 실험에서 action target과 benchmark 실행 인터페이스를 어떻게 맞추는지 정리한 파일이다.

관련 핵심 코드:
- [utils.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/utils.py)
- [dataset_loader.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/dataset/dataset_loader.py)
- [dataset_config.yaml](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/dataset_config.yaml)
- [JV direct template](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/action_pipeline_joint_velocity_direct_template.yaml)
- [JP -> benchmark JV servo template](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/action_pipeline_joint_position_to_benchmark_joint_velocity_servo_template.yaml)
- [JP direct template](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/action_pipeline_joint_position_direct_template.yaml)

## RLBench / Colosseum의 공식 JV 기본 동작

공식 `JointVelocity`는 별도 `gain`, `clip`, `servo_steps` 같은 기본값이 없다.

실제 구현:
- [arm_action_modes.py](/home/cvlab-dgx/siwon/object_centric_diffusion/third_party/RLBench/rlbench/action_modes/arm_action_modes.py#L57)

동작은 아래와 같다.
1. `set_joint_target_velocities(action)`
2. `scene.step()`
3. `set_joint_target_velocities(zeros)`

control mode도 아래처럼 바뀐다.
- control loop `False`
- motor locked at zero velocity `True`
- [arm_action_modes.py](/home/cvlab-dgx/siwon/object_centric_diffusion/third_party/RLBench/rlbench/action_modes/arm_action_modes.py#L66)

즉 RLBench/Colosseum의 공식 JV는 `servo controller`가 아니라 `1-step velocity pulse`다.

## Preset 목록

### 1. `joint_position_to_benchmark_joint_velocity_servo`

의미:
- 학습 target은 `next joint_positions + gripper`
- benchmark 실행은 `JointVelocity + gripper`
- 중간에 `joint_position -> joint_velocity servo adapter`를 거친다

기본값:
- `joint_velocity_servo_gain: 20.0`
- `joint_velocity_servo_clip: 1.0`
- `joint_velocity_servo_steps: 2`
- `joint_velocity_servo_tolerance: 0.01`

식:
```text
v = clip(gain * (q_target - q_current), -clip, clip)
```

언제 쓰나:
- benchmark interface는 JV로 유지해야 하는데
- policy/replay target은 JP가 더 안정적일 때

### 2. `joint_velocity_direct`

의미:
- 학습 target은 `obs.joint_velocities + next gripper`
- 실행도 RLBench/Colosseum 공식 `JointVelocity + gripper`
- 어댑터 없음

공식 baseline에 가장 가까운 모드다.

언제 쓰나:
- RLBench/Colosseum의 원래 interface를 그대로 따라갈 때
- ELSA 기본 baseline과 최대한 맞추고 싶을 때

### 3. `joint_position_direct`

의미:
- 학습 target은 `next joint_positions + gripper`
- 실행은 `JointPosition(absolute) + gripper`
- 어댑터 없음

언제 쓰나:
- replay upper bound를 먼저 확인하고 싶을 때
- benchmark JV보다 JP direct가 더 안정적인지 볼 때

## Config 사용법

`dataset_config.yaml`이나 실험 YAML에서 아래처럼 고르면 된다.

### A. JP 학습 -> benchmark JV servo 실행

```yaml
dataset:
  action_pipeline_preset: joint_position_to_benchmark_joint_velocity_servo
  action_representation: auto
  execution_action_interface: auto
  execution_action_adapter: auto
  joint_velocity_servo_gain: 20.0
  joint_velocity_servo_clip: 1.0
  joint_velocity_servo_steps: 2
  joint_velocity_servo_tolerance: 0.01
```

주의:
- JP 계열 preset은 `transform.action_min/max`가 joint-position bounds여야 한다.
- 바로 쓰려면 위 template YAML을 복사하는 게 가장 안전하다.

### B. JV 학습 -> JV direct 실행

```yaml
dataset:
  action_pipeline_preset: joint_velocity_direct
  action_representation: auto
  execution_action_interface: auto
  execution_action_adapter: auto
```

### C. JP 학습 -> JP direct 실행

```yaml
dataset:
  action_pipeline_preset: joint_position_direct
  action_representation: auto
  execution_action_interface: auto
  execution_action_adapter: auto
```

주의:
- 이 모드도 JP bounds를 써야 한다.

## 명시적 override 규칙

preset은 기본값만 채운다.

즉 아래처럼 explicit field를 주면 preset보다 explicit 값이 우선한다.

```yaml
dataset:
  action_pipeline_preset: joint_position_to_benchmark_joint_velocity_servo
  execution_action_interface: joint_velocity
  execution_action_adapter: joint_position_to_joint_velocity_servo
  joint_velocity_servo_gain: 10.0
```

## 현재 추천

공통 replay ceiling 기준:
- direct JP는 `close 1.0 / slide 1.0 / insert 0.8 / scoop 0.9`
- benchmark JV servo `g20/c1.0/s2`는 `close 0.9 / slide 1.0 / insert 0.9 / scoop 0.9`

즉:
- replay upper bound 기준 범용성은 `joint_position_direct`가 아직 더 안전하다
- benchmark JV strict 경로가 필요하면 `joint_position_to_benchmark_joint_velocity_servo` + `g20/c1.0/s2`가 현재 단일 후보 중 가장 균형적이다
- 공식 baseline 비교가 핵심이면 `joint_velocity_direct`를 써야 한다
