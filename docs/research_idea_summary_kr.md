# 연구 아이디어 정리: FLAME 위에서 Federated Robotic Manipulation 개선

> 최종 갱신: 2026-04-28
> 관련 repo: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge`

---

## 1. 배경: FLAME 논문이 보여준 것과 남긴 문제

### 1.1 FLAME이 한 일

FLAME (Federated Learning Across Manipulation Environments)은 로봇 매니퓰레이션을 위한 **최초의 federated learning 벤치마크**다.

- **데이터**: 4개 태스크 × 420개 환경 × 100개 demo = 약 160,000 에피소드 (15M+ 샘플)
- **환경 이질성**: 배경 텍스처, 조명 색상, 카메라 시점, 오브젝트 크기/색상/텍스처 등 perturbation factor가 환경마다 다름
- **프레임워크**: FLOWER 기반 FL 파이프라인, 각 클라이언트가 고유 환경에서 local 학습 후 global model 집계
- **베이스라인 정책**: 3-layer CNN + MLP (BCPolicy), MSE loss, joint velocity 출력
- **FL 방법 비교**: FedAvg, FedAvgM, FedOpt, Krum

### 1.2 FLAME 결과가 드러낸 한계

| 태스크 | FedAvg SR | FedOpt SR | 비고 |
|---|---:|---:|---|
| Close Box | **84%** | 70% | 유일하게 의미 있는 성공률 |
| Slide Block | 24% | 28% | 낮은 성공률 |
| Insert Peg | 0% | 0% | 완전 실패 |
| Scoop | 0% | 0% | 완전 실패 |

**핵심 문제**: CNN 기반 단순 정책 + 기본 FL 알고리즘으로는 시각적 이질성이 큰 cross-environment 일반화가 되지 않는다.

### 1.3 논문이 직접 지적한 gap

- offline evaluation (RMSE)과 online evaluation (success rate)이 일치하지 않음
- 대부분의 FL 방법이 Close Box에만 효과적이고 난이도 높은 태스크에서 전면 실패
- 조명/시점/텍스처에 robust한 시각 표현과 manipulation-specific FL 전략이 필요

---

## 2. 우리의 핵심 가설

> **Foundation vision model + Diffusion policy head + FL-aware 경량 적응**의 조합이
> FLAME 벤치마크에서 cross-environment 일반화를 근본적으로 개선할 수 있다.

구체적으로 세 축에서 FLAME 베이스라인을 교체한다:

| 축 | FLAME 원본 | 우리의 제안 |
|---|---|---|
| **시각 인코더** | 3-layer CNN (scratch) | Frozen DINOv3 + LoRA + Depth Anything |
| **정책 헤드** | Deterministic MLP | Diffusion-based denoising head |
| **FL 전략** | 기본 FedAvg/FedOpt | Frozen backbone + LoRA-only aggregation |

---

## 3. 방법론 상세

### 3.1 시각 인코더: Frozen DINOv3 + LoRA + Depth Anything

**왜 foundation model인가:**
- CNN은 환경마다 다른 텍스처/조명/시점에 쉽게 오버핏됨
- DINOv3 (ViT-S/16)는 self-supervised pre-training으로 시각적 변이에 이미 robust한 feature를 가짐
- backbone을 frozen으로 두면 FL에서 통신량과 client drift가 동시에 줄어듦

**LoRA 적응:**
- backbone의 마지막 N개 attention block에만 low-rank adapter 삽입
- 실험에서 rank=4~8, 마지막 8 block이 sweet spot
- global aggregation 대상은 LoRA weight + projector + policy head뿐 → 통신 효율적

**Depth Anything 분기:**
- monocular depth estimation 모델 (Depth Anything Small)을 frozen으로 추가
- DINO feature와 depth feature를 concat하여 구조적 깊이 정보 보강
- 시점 변화에 2D feature만으로는 잡기 어려운 3D geometry cue 제공

**Proprioceptive-Visual Fusion:**
- joint position (proprio) 정보를 visual feature에 FiLM 방식으로 주입
- `gated_global_film`: proprio → (gamma, beta) 예측 후 gate 함수로 기여도 조절
- 로봇 상태가 시각 해석을 조건화하여, 같은 장면이라도 로봇 자세에 따라 다르게 해석

### 3.2 정책 헤드: Diffusion Policy

**왜 diffusion인가:**
- 매니퓰레이션 데모는 본질적으로 multi-modal (같은 상태에서 여러 유효 행동)
- MSE loss의 deterministic MLP는 mode averaging → 부정확한 중간값 예측
- diffusion head는 조건부 분포를 모델링하여 더 정확한 행동 시퀀스 생성

**구현:**
- DDPM 기반 denoising MLP, 20-step schedule
- context = [visual_embedding ; state_embedding] → FC → 512-dim
- action noise prediction: noise MLP conditioned on (noisy_action, context, timestep)
- inference: reverse diffusion으로 action sampling 후 [-1, 1] clamp

**Action chunking:**
- `chunk_len=4, execute_steps=2`: 4-step 행동 시퀀스 예측, 2 step 실행 후 재계획
- receding horizon 방식으로 closed-loop 안정성 확보

**Separate gripper head:**
- 연속 arm action (diffusion)과 이산 gripper action (binary classification) 분리
- arm 7-DoF는 diffusion, gripper는 sigmoid + BCE loss
- 그리퍼의 open/close 결정은 multi-modal이 아니라 deterministic이므로 분리가 합리적

### 3.3 FL 전략: Frozen Backbone + LoRA Aggregation

**핵심 아이디어:**
- DINOv3 base weight는 모든 클라이언트에서 동일하게 frozen → drift 없음
- aggregation 대상: LoRA weights + projector + MLP encoder + diffusion head
- 총 trainable parameter가 전체 모델의 ~10% 수준 → 통신량 대폭 절감

**FL 방법:**
- 현재 기준 FedAvg가 FedProx보다 우수 (FedProx의 prox penalty가 diffusion + LoRA에서 오히려 방해)
- 후속 실험 후보: SCAFFOLD (client drift correction)

---

## 4. 실험 설계와 현재까지의 결과

### 4.1 실험 파이프라인

```
Stage 0: Replay upper bound 확인
    → action interface별 dataset replay ceiling 측정
    
Stage 1: Same-env single task (env0)
    → vision × action × policy head 조합 탐색
    → 4개 태스크 각각에서 best config 확인
    
Stage 2: Same-env env sweep (env0~3)
    → best config의 환경 간 편차 측정
    
Stage 3: Same-env multi-env (env0~4 동시 학습)
    → 단일 모델이 여러 환경을 커버하는지 확인 ← 현재 단계
    
Stage 4: Federated pilot
    → FL 파이프라인에서 동일 config의 성능 확인
    
Stage 5: FL 전략 비교 & ablation
    → FedAvg vs SCAFFOLD vs FedBN 등
```

### 4.2 Same-env 주요 결과 (slide_block_to_target, env0, 50ep)

| 설정 | SR | RMSE |
|---|---:|---:|
| FLAME 원본 (CNN + MLP + FedAvg) | 0.24 | - |
| CNN + JV + AdaLN | 0.40 | 0.40 |
| DINO frozen + JV + MLP | 0.40 | 0.40 |
| DINO LoRA4 + JV + Diffusion | **0.70** | 0.10 |
| DINO LoRA8 + JV + Diffusion | **0.70** | 0.10 |
| DINO+Depth LoRA4 + JV + Diffusion | 0.60 | 0.11 |
| DINO+Depth LoRA8 + JV + Diffusion | **0.75** | 0.10 |

### 4.3 4-task same-env 결과 (JV + DINO LoRA4 + Diffusion, 50ep)

| 태스크 | SR |
|---|---:|
| slide_block_to_target | **0.70** |
| close_box | 0.20 |
| scoop_with_spatula | 0.05 |
| insert_onto_square_peg | 0.00 |

### 4.4 Federated pilot 결과 (DINO LoRA4 + Diffusion + JV, 10 rounds)

| 태스크 | FedAvg | FedProx |
|---|---:|---:|
| close_box | **0.34** | 0.22 |
| slide_block_to_target | **0.16** | 0.04 |

- diffusion head가 FL 환경에서도 동작함을 확인
- 현재 설정에서 FedAvg > FedProx

---

## 5. 현재 진행 중인 실험 (2026-04-27~)

### 5.1 Multi-env close_box ablation

동일 태스크(close_box)에 대해 여러 환경(env0~4)의 데이터를 한 모델로 학습:

| 실험 | 핵심 차이 | GPU |
|---|---|---|
| `chunk4exec2` | action chunking (4-step predict, 2-step execute) | cuda:0 |
| `splitgripper_w1` | arm/gripper 분리 head | cuda:0 |
| `chunk4exec2_splitgripper` | chunking + gripper 분리 조합 | cuda:0 |

목적: multi-env 학습에서 action formulation과 gripper head 분리의 영향 측정

### 5.2 Proprio-visual fusion sweep (완료)

| fusion mode | 태스크 | 목적 |
|---|---|---|
| `gated_global_film` | close_box, slide | proprio가 visual feature를 gate로 조절 |
| `global_film (weak scale=0.1)` | close_box, slide | 약한 FiLM conditioning |

---

## 6. 앞으로의 실험 로드맵

### Phase 1: Same-env 완성 (현재 ~ 1주)

- [ ] multi-env close_box ablation 결과 수집
- [ ] multi-env 4-task sweep (best config로 slide, insert, scoop 추가)
- [ ] env0~4 → env0~9 확장하여 generalization 곡선 측정

### Phase 2: Federated 본실험 (1~2주)

- [ ] best same-env config를 FL pipeline에 투입
- [ ] FedAvg baseline: rounds=20, local_epochs=5, fraction_fit=0.05
- [ ] SCAFFOLD 구현 및 비교
- [ ] local epoch 수 ablation (1, 5, 10, 25)
- [ ] LoRA rank ablation in FL (2, 4, 8)

### Phase 3: 구조 확장 (2~3주)

- [ ] VolumeDP-lite: 3D voxel lifting + spatial token + multi-token diffusion decoder
- [ ] FKCameraObjectPolicy: FK-conditioned view adaptor + object-centric query tokens
- [ ] camera intrinsics/extrinsics를 dataset loader에 추가

### Phase 4: 논문용 최종 실험

- [ ] 4-task × 5-method 비교 테이블 (FLAME baseline, ours-same-env, ours-FL 등)
- [ ] env 수 scaling 실험 (5, 20, 50, 100, 400 clients)
- [ ] ablation table: vision encoder / policy head / LoRA / fusion mode

---

## 7. 예상 contribution 정리

### C1. Foundation Model 기반 FL-friendly 시각 인코더
- frozen DINOv3 + LoRA: FL에서 backbone drift 제거, 통신량 ~90% 절감
- Depth Anything concat: monocular depth prior로 시점 변화에 대한 robustness 향상

### C2. Diffusion Policy Head for Federated Manipulation
- multi-modal action 분포 모델링으로 deterministic MLP 대비 성능 대폭 개선
- FL에서도 diffusion head가 동작함을 실험적으로 검증

### C3. Federated-aware 경량 적응 전략
- LoRA-only aggregation: client drift 최소화와 통신 효율의 동시 달성
- separate gripper head: arm continuous / gripper discrete 분리로 정밀도 향상
- proprio-visual fusion: 로봇 상태 기반 시각 조건화로 cross-environment 일반화 보조

### C4 (후보). 3D-aware Spatial Token for FL
- VolumeDP-lite: camera-aligned voxel representation으로 시점 정규화
- 서로 다른 카메라 설정의 클라이언트 간 feature 정렬 가능성

---

## 8. 한 문장 요약

> FLAME 벤치마크의 CNN+MLP 베이스라인을 **frozen DINOv3 + LoRA + Diffusion Policy**로 교체하고,
> **LoRA-only FL aggregation**을 통해 시각적 이질성이 큰 다환경 로봇 매니퓰레이션에서
> federated learning의 실질적 성능 향상을 달성한다.

---

## 부록 A: 코드 구조

| 파일 | 역할 |
|---|---|
| `elsa_learning_agent/agent.py` | BCPolicy, DINOv3 encoder, LoRA, Depth Anything, VolumeDP-lite, Diffusion head 전체 구현 |
| `scripts/train_same_env_bcpolicy_probe.py` | same-env 학습/평가 스크립트 |
| `federated_elsa_robotics/task.py` | FL client 학습 루프 |
| `federated_elsa_robotics/strategies.py` | FL server aggregation 전략 |
| `experiments/*.yaml` | 실험별 config (vision, action, FL 설정) |

## 부록 B: 주요 참고 논문

- **FLAME**: Bou Betran et al., "FLAME: A Federated Learning Benchmark for Robotic Manipulation", arXiv:2503.01729, 2025
- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
- **VolumeDP**: "VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning", arXiv:2603.17720
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
- **Depth Anything**: Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data", CVPR 2024
