# Federated Method Plan

이 문서는 현재 ELSA/FLAME 코드베이스에서 현실적으로 밀어야 할 FL 방법을 정리한 파일이다.

## 현재 파이프라인 해석

현재 Flower 파이프라인은 아래 구조다.
- client: local dataset shard 로딩 + local SGD
- server: trainable parameter aggregation

관련 코드:
- [client_app.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/client_app.py)
- [server_app.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/server_app.py)
- [task.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/task.py)

즉 지금도 client가 local training은 하고 있고, `federated`의 핵심은 server aggregation 방식과 local objective에 있다.

## 핵심 관찰

1. FLAME/Colosseum variation은 색상 변화, 배경 변화, 카메라 시점 변화가 크다.
2. 평가는 결국 `single global model` 기준으로 보는 쪽이어서, 지나친 client-personalization은 바로 쓰기 어렵다.
3. 따라서 지금 코드베이스에서는 `client별 비공유 모듈`보다 `global shared model + drift 억제 + 강한 시각 backbone`이 더 현실적이다.

## 권장 방법

### 1. 기본 FL 방법

기본은 아래로 두는 게 맞다.
- action: `chunk3 + execute2`
- vision: `frozen DINO`
- FL regularizer: `FedProx`

이유:
- `chunk3 + execute2`가 현재 FL 일반화에서 제일 나은 신호를 냈다.
- `frozen DINO`는 색/시점 변화에 대해 CNN보다 안정적이다.
- `FedProx`는 heterogeneous client에서 local drift를 줄이는 가장 싼 방법이다.

### 2. local epoch / round

환경 차이가 큰데 local epoch을 너무 키우면 각 client가 자기 도메인으로 과적합한다.

추천:
- local epoch: `5 ~ 10`
- rounds: `20 ~ 30`

지금처럼 `local 50`은 CPU/GPU 비용도 크고 drift도 커진다.

### 3. aggregation 범위

현재 구조에서는 `trainable-only aggregation`을 유지하는 게 맞다.

즉:
- frozen backbone은 그대로 둠
- projector + policy head만 aggregation

이건 이미 코드가 그 방향이다.
- [task.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/task.py#L49)

### 4. client personalization은?

현재 benchmark 평가가 global model 중심이면, 아래는 우선순위가 낮다.
- client-specific BN 비공유
- client-specific adapter만 유지

이유:
- test env에는 그 client state를 그대로 가져갈 수 없기 때문이다.

즉 지금은 personalization보다
- stronger global visual encoder
- local drift regularization
- action formulation 개선
이 우선이다.

## 추천 실험 순서

### A. CPU smoke

목적:
- 코드 경로 확인
- dataset/action/eval mismatch 제거

설정:
- `rounds=2`
- `local_epochs=2`
- `fraction_fit=0.02`
- `prox_mu=1e-3`

스크립트:
- [run_flower_programmatic_one_task_cpu.sh](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_flower_programmatic_one_task_cpu.sh)

### B. FL main baseline

설정:
- `chunk3 + execute2`
- `frozen DINO`
- `FedProx`
- `rounds=20`
- `local_epochs=5`

### C. same-env sanity

FL이 안 될 때는 먼저 same-env가 되는지 본다.
- same-env에서 안 되면 FL 문제가 아니라 policy/closed-loop 문제다.

## 색상 변화 / 카메라 시점 변화 대응

가장 현실적인 대응:
- frozen DINO backbone
- local training에 color jitter / brightness / contrast augment 강화
- local epoch 축소 + FedProx

우선순위가 낮은 것:
- 복잡한 client-specific feature alignment
- non-shared modules를 많이 두는 personalization

이유:
- 현재는 single global model 성능을 올리는 게 먼저다.

## 추천 결론

현재 코드베이스 기준으로는 아래가 메인 baseline이다.

- `action_pipeline_preset: joint_position_to_benchmark_joint_velocity_servo`
- `action_chunk_len: 3`
- `receding_horizon_execute_steps: 2`
- `vision_backbone: dinov3_vits16_frozen`
- `prox_mu: 1e-3`
- `local_epochs: 5`
- `rounds: 20`

만약 benchmark strict JV보다 replay ceiling을 먼저 안정화하고 싶으면:
- `action_pipeline_preset: joint_position_direct`

공식 baseline 재현 비교가 목적이면:
- `action_pipeline_preset: joint_velocity_direct`
