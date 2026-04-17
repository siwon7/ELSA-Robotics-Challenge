# FL Experiment Roadmap

이 문서는 현재 ELSA/FLAME 코드베이스에서 `federated learning` 관점으로 실험을 어떻게 구성하는 게 맞는지 정리한 파일이다.

## 1. 현재 코드가 실제로 하는 것

현재 server 쪽 전략은 `FedAvg`이고, client local objective에 `FedProx` penalty를 추가하는 구조다.

관련 코드:
- [server_app.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/server_app.py)
- [strategies.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/strategies.py)
- [task.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/task.py)

즉 정확히 말하면:
- server aggregation: `FedAvg`
- client local regularization: `FedProx`

이 구조는 heterogeneous client drift를 줄이기 위한 가장 싼 baseline이다.

## 2. 현재 환경에서 중요한 이질성

현재 benchmark에서 큰 variation은 대략 아래 세 축이다.
- 색상 변화
- 배경/조명 변화
- 카메라 시점 변화

RLBench/Colosseum 쪽 참고:
- camera pose variation: `/tmp/robot-colosseum/colosseum/variations/camera_pose.py`
- background variation: `/tmp/robot-colosseum/colosseum/variations/background_texture.py`
- object color variation: `/tmp/robot-colosseum/colosseum/variations/object_color.py`
- light color variation: `/tmp/robot-colosseum/colosseum/variations/light_color.py`

이건 전형적인 non-IID visual domain shift다.

## 3. 왜 personalization을 바로 밀지 않는가

현재 평가는 결국 `single global model` 성능이 중요하다.

그래서 아래는 지금 바로 우선순위가 높지 않다.
- client별 BN 통계만 유지하는 강한 personalization
- client별 adapter를 따로 유지하는 구조
- client마다 비공유 visual head를 두는 구조

이유:
- test 시점에 client identity가 명확하지 않거나
- global model 하나로 평가해야 하는 경우
- personalization 이득을 그대로 가져가기 어렵다.

즉 지금은 `client마다 다르게 맞추는 것`보다 `global model이 덜 망가지게 학습하는 것`이 먼저다.

## 4. 현재 추천하는 실험 축

### A. Action

현재 FL 기준으로 제일 신호가 있었던 건:
- `chunk3 + execute_steps=2`

이유:
- one-step `AbsJointPos`는 replay ceiling은 높았지만 learned policy가 rollout에서 무너졌다.
- `chunk4`는 same-env에서는 더 나았지만 FL generalization은 `chunk3`가 더 안정적이었다.

즉 FL 메인 baseline은:
- `action_chunk_len: 3`
- `receding_horizon_execute_steps: 2`

### B. Vision

현재 가장 안전한 visual encoder는:
- `frozen DINO`

이유:
- CNN보다 color/viewpoint variation에 덜 취약하다.
- backbone을 frozen으로 두면 FL 통신량과 client drift를 줄일 수 있다.
- projector + policy head만 aggregation 하면 충분히 가볍다.

### C. FL optimizer / strategy

현재 우선순위는 아래 순서가 맞다.

1. `FedAvg + local FedProx`
2. `SCAFFOLD`
3. `FedNova`
4. `FedBN`류 personalization

해석:
- `FedProx`: 구현 비용이 가장 낮고 지금 코드에도 이미 맞아 있다.
- `SCAFFOLD`: client drift가 심할 때 다음으로 가장 말이 되는 대안이다.
- `FedNova`: local update 길이 차이 정규화에는 좋지만 현재 병목과는 한 단계 떨어진다.
- `FedBN`: visual heterogeneity에는 맞을 수 있지만 single-global-model 평가와 충돌할 수 있다.

즉 지금 상황에서 `FedProx보다 좋은가?`에 대한 가장 강한 다음 후보는:
- `SCAFFOLD`

단, 현재 repo에는 아직 clean implementation이 없다.

## 5. Depth Anything + DINO는 언제 넣는가

`Depth Anything + DINO`는 만들 수는 있다. 하지만 우선순위는 두 번째다.

먼저 해야 할 것:
- `chunk3 + execute2 + frozen DINO + FedProx`

그 다음 볼 것:
- `chunk3 + execute2 + frozen DINO + Depth Anything + FedProx`

이유:
- 지금 문제는 visual encoder만의 문제가 아니라 FL drift와 closed-loop behavior가 함께 엮여 있다.
- depth branch까지 동시에 넣으면 원인 분리가 어려워진다.

따라서 추천 순서는:
1. DINO만으로 FL baseline 고정
2. 그 baseline이 어느 정도 살아난 뒤 depth branch 추가

## 6. 카메라 shift 실험은 현재 어떤 상태인가

현재 repo에 남아 있는 건 `FKCameraObjectPolicy_method.txt`다.
- [FKCameraObjectPolicy_method.txt](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/FKCameraObjectPolicy_method.txt)

이 파일은:
- frozen DINOv2 계열 backbone
- LoRA
- FK-conditioned camera adaptor
- latent camera shift
- robot/object token extraction

같은 아이디어를 정리한 method note다.

중요한 점:
- 이건 **현재 repo에서 검증된 실험 결과 파일이 아니다**
- 즉 `camera shift baseline이 이미 구현돼서 성능이 나왔다`고 보면 안 된다
- 지금 상태에선 **방법 요약 문서**에 가깝다

따라서 camera shift는 현재 기준으로:
- `existing validated baseline`: 아님
- `future method candidate`: 맞음

## 7. 지금 당장 가장 합리적인 메인 baseline

현재 추천 메인 baseline:
- action: `chunk3`
- execution: `execute_steps=2`
- visual encoder: `frozen DINO`
- FL: `FedAvg + local FedProx`
- local epochs: `5`
- rounds: `20`

이걸 config 식으로 쓰면:

```yaml
dataset:
  action_chunk_len: 3
  receding_horizon_execute_steps: 2
  action_pipeline_preset: joint_position_to_benchmark_joint_velocity_servo

model:
  vision_backbone: dinov3_vits16_frozen

runtime:
  prox_mu: 0.001
```

권장 학습량:
- `local_epochs = 5`
- `rounds = 20`
- `fraction_fit = 0.05`

이유:
- 지금처럼 `local 50`은 CPU/GPU 비용이 너무 크고 client drift도 커진다.

## 8. CPU 기준 추천 실험 순서

GPU를 많이 쓰는 상태라면 CPU로 smoke를 먼저 보는 게 맞다.

스크립트:
- [run_flower_programmatic_one_task_cpu.sh](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_flower_programmatic_one_task_cpu.sh)

### Step 1. CPU smoke

목적:
- config / dataset / eval path 확인
- action preset mismatch 제거

추천:
- `rounds=2`
- `local_epochs=2`
- `fraction_fit=0.02`

예시:

```bash
scripts/run_flower_programmatic_one_task_cpu.sh \
  slide_block_to_target 2 2 cpu-smoke-v1 0.02 0.9 0.001
```

### Step 2. CPU same-env sanity

목적:
- FL 이전에 same-env closed-loop가 되는지 확인

기준:
- same-env에서도 안 되면 FL 문제가 아니다

### Step 3. GPU full run

CPU smoke가 통과한 뒤에만 GPU full run:
- `rounds=20`
- `local_epochs=5`
- `fraction_fit=0.05`

## 9. 다음 구현 우선순위

지금 코드 기준 다음 우선순위는:

1. `chunk3 + execute2 + frozen DINO + FedProx`를 CPU smoke -> GPU full로 고정
2. `SCAFFOLD` server/client update 추가
3. `DINO + Depth Anything` 결합
4. `FKCameraObjectPolicy` 같은 camera-adaptor 계열은 그 다음

즉 camera shift 대응을 바로 크게 설계하기보다:
- 먼저 global FL baseline을 안정화
- 그다음 camera-adaptor 실험을 올리는 순서가 맞다

## 10. 짧은 결론

현재 상황에서 가장 합리적인 FL 실험 구성은:
- `chunk3 + execute2`
- `frozen DINO`
- `FedAvg + local FedProx`
- `local_epochs=5`
- `rounds=20`

그리고 `camera shift` 실험은 현재 validated baseline이 아니라 method note 단계다.
다음으로 진짜 추가할 가치가 큰 FL 방법은 `SCAFFOLD`다.
