# IC613 Federated Learning 강의 자료 분석과 ELSA 적용안

작성일: 2026-05-07 KST
대상 자료: `IC613 lec01 Intro.7z`
작업 위치: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/fede`

## 0. 작업 방식

- `harness-100`의 `63-research-assistant` 하네스를 `fede/.claude`에 적용했다.
- 압축 파일은 PPT가 아니라 PDF 24개로 구성되어 있었다.
- 원본 PDF는 `fede/course_materials/ic613_lectures/`에 풀었다.
- 모든 PDF에 대해 `pdftotext -layout`으로 텍스트를 추출했고,
  inventory는 `fede/extracted_text/pdf_inventory.tsv`에 저장했다.
- 관련 구현 repo는 `fede/repos/`에 shallow/partial clone으로 받았다.
- ELSA에 옮겨 쓸 수 있는 hook 초안은 `fede/modules/`에 모았다.

## 1. 전체 강의 흐름

IC613 자료는 단순히 FedAvg/FedProx 사용법을 가르치는 수업이 아니라, 다음 질문을
순서대로 좁혀 간다.

1. SGD가 왜 수렴하고, learning rate/batch/variance가 왜 중요한가.
2. FL에서 local update가 많아질 때 왜 FedAvg가 client drift를 만든는가.
3. 그 drift가 statistical heterogeneity, computational heterogeneity, partial
   participation 중 어디서 생기는가.
4. FedProx, SCAFFOLD, FedNova, FedExp 같은 방법이 각각 어떤 drift를 줄이는가.
5. 평균 성능이 아니라 worst-client/fairness/incentive/personalization/privacy까지
   고려하면 objective 자체가 어떻게 바뀌는가.

우리 ELSA/FLAME 연구에 가장 중요한 결론은 다음이다.

> FL 알고리즘을 바꾸기 전에 heterogeneity의 종류를 먼저 분리해야 한다.
> ELSA에서는 vision shift, camera geometry shift, action/controller timing shift,
> gripper event shift가 섞여 있으므로, FedAvg가 나쁘다는 사실만으로 SCAFFOLD나
> FedProx가 답이라고 말할 수 없다.

## 2. 자료 인벤토리

| 그룹 | 파일 수 | 내용 |
|---|---:|---|
| Lecture slides | 15 | SGD 기초, FL intro, heterogeneity, partial selection, fairness, personalization, privacy |
| Attached papers | 9 | AFL, q-FFL, SCAFFOLD, FedProx, FedNova, Power-of-Choice, Ditto, FedExp, MaxFL |
| 총 페이지 | 895 | PDF metadata 기준 |

## 3. 강의별 핵심과 ELSA 적용

### Lec01 - Introduction

강의가 보는 FL의 기본 동기는 edge/client에 데이터가 있고, raw data 이동이 비싸거나
privacy/legal 측면에서 불가능하다는 것이다. 또한 distributed ML과 FL을 구분한다.
distributed ML은 보통 IID shard와 빠른 동기화가 가능하다고 가정하지만, FL은 client
data, compute, availability가 모두 불균일하다.

ELSA 적용:

- FLAME/ELSA는 전형적인 cross-silo 또는 simulated cross-device FL이다.
- 우리 데이터는 privacy보다 "환경/client마다 다른 visual/camera/action distribution"이
  핵심이다.
- 논문 framing은 privacy-first보다 heterogeneity-first가 맞다.

### Lec02-Lec04 - SGD, convergence, variance reduction

SGD/mini-batch/GD 수렴 분석을 통해 learning rate, batch variance, smoothness,
strong convexity/non-convex stationary point, SAG/SAGA/SVRG 같은 variance reduction을
정리한다. 이후 FL 방법들은 대부분 이 분석 template의 변형이다.

ELSA 적용:

- diffusion loss는 stochastic timestep/noise sampling 때문에 같은 batch에서도 variance가
  크다. FL에서는 client variance와 diffusion stochastic variance가 겹친다.
- local epoch를 크게 잡으면 communication은 줄지만 client drift가 커진다.
- TGAC처럼 gripper BCE, EE aux loss, diffusion MSE를 섞으면 loss scale이 바뀐다.
  FedProx `mu`는 loss scale과 같이 해석해야 한다.
- 지금 FL sweep은 `local_epochs={1,3,5}`부터 시작해야 한다. 10 이상은 원인 분리가
  끝난 뒤에 둔다.

### Lec05 - Federated Learning Intro

FedAvg framework, global/local objective, client fraction `C`, local epochs `E`,
mini-batch `B`의 효과를 소개한다. FedAvg는 가장 단순한 baseline이지만 data
heterogeneity와 partial participation에 민감하다.

ELSA 적용:

- 현재 ELSA Flower 구현은 `FedAvg` 기본선에 더해 `fedexp`, `fednova`, `qfedavg`, `afl`, `maxfl` server aggregation과 FedPer-style local head surface를 선택할 수 있다.
- DINO/Depth base를 frozen으로 두고 LoRA/projector/diffusion decoder만 aggregate하는
  현재 방향은 FedAvg를 약하게 만드는 heterogeneity surface를 줄이는 전략이다.
- FedAvg를 반드시 남겨야 한다. 기존 pilot에서도 `DINO LoRA4 + diffusion + JV direct`는
  FedAvg가 FedProx보다 좋았다.

### Lec06 - Heterogeneity 1: FedProx, SCAFFOLD, FedExp

statistical heterogeneity 때문에 client local optimum이 global optimum과 다르고,
local update가 client 방향으로 drift한다. FedProx는 local objective에 proximal term을
넣어 global model에서 멀어지는 것을 막고, SCAFFOLD는 server/client control variate로
client drift를 보정한다. FedExp는 server learning rate adaptation 관점으로 FedAvg를
가속한다.

ELSA 적용:

- FedProx는 이미 `federated_elsa_robotics/task.py`에 들어가 있다.
- `mu=1e-3` 고정은 위험하다. 강의/논문 모두 tuning을 요구한다.
- SCAFFOLD는 의미 있지만 stateful control variate가 필요해 구현비가 크다.
- FedExp/FedExP류 server LR ablation은 server-only라 구현 난도가 낮다. 다음 코드
  우선순위는 SCAFFOLD보다 bounded server LR이 더 현실적이다.

### Lec07 - Heterogeneity 2: computational heterogeneity, FedNova

client마다 local update 수가 다르면 FedAvg/FedProx가 의도한 global objective와 다른
stationary point로 갈 수 있다. FedNova는 update 길이를 정규화해서 objective
inconsistency를 줄인다.

ELSA 적용:

- 현재 우리 queue는 local epochs를 동일하게 맞추는 편이므로 FedNova 우선순위는 낮다.
- 하지만 VolumeDP-full은 client별 throughput 차이가 커질 수 있다. power issue나
  GPU scheduling 때문에 어떤 client는 더 적은 step을 돌 수 있다면 FedNova가 필요해진다.
- 구현하려면 client가 `local_steps`, sample count, normalized delta 정보를 반환해야 한다.

### Lec08-Lec09 - Partial selection, FedVARP, Power-of-Choice

매 round 모든 client가 참여하지 않으면 participation variance가 생긴다. FedVARP는
SAGA식으로 client participation variance를 줄이고, Power-of-Choice는 candidate 중
loss가 큰 client를 선택해 더 빠르게 어려운 client를 학습한다.

ELSA 적용:

- FLAME 환경에서 random env/client sampling만 쓰면 rare failure mode를 늦게 본다.
- high-loss/high-failure env를 일부러 뽑는 diagnostic round는 gripper transition,
  camera shift, close/scoop failure mode를 빨리 드러낸다.
- 다만 biased selection은 평가도 같이 bias시킬 수 있다. paper table에는 random-client
  mean SR, worst-env SR, selected-client SR을 분리해서 보고해야 한다.

### Lec10 - Fairness, AFL, q-FFL

AFL은 가능한 client mixture 중 worst-case distribution에 대해 잘하는 global model을
목표로 한다. q-FFL은 loss가 큰 client에 더 큰 가중을 주어 average accuracy와
device-level fairness 사이를 조절한다.

ELSA 적용:

- 우리 논문에서는 mean SR만 보면 안 된다. FLAME/robotics에서는 특정 env/camera/task가
  계속 실패하면 global policy로 보기 어렵다.
- q-FFL식 client weighting은 "worst-camera/worst-env improvement" ablation으로 좋다.
- 단, high-loss client가 noisy simulator, broken data, action contract mismatch라면
  q-FFL은 오히려 잘못된 client를 과대반영한다. 먼저 data/rollout hygiene filter가 필요하다.

### Lec11-Lec13 - Personalization, Ditto, clustering, meta-learning, MTL, model merging

personalized FL은 global-only와 local-only 사이를 다룬다. Ditto는 global model과
personalized local model 사이에 proximal regularization을 둔다. clustering은 client를
cohort로 묶고, meta-learning은 빠른 local adaptation을 목표로 한다. MTL은 shared trunk와
client-specific head, MoE, soft-parameter sharing, multi-objective optimization으로 확장된다.
model merging/task arithmetic은 여러 fine-tuned model의 차이를 조합하는 관점이다.

ELSA 적용:

- FLAME 평가가 single global checkpoint라면 personalization은 main claim이 되기 어렵다.
- 그러나 analysis로는 매우 좋다. camera cohort, task cohort, gripper timing cohort를
  찾아 "왜 global averaging이 실패했는가"를 설명할 수 있다.
- 가장 현실적인 personalized ablation:
  - shared DINO/LoRA/VolumeDP representation + global diffusion decoder
  - local gripper calibration 또는 local tiny action head
  - task/camera cohort별 FedAvg
- VolumeDP에서는 camera-aware voxel module을 local로 둘지, global로 둘지가 중요한
  personalization/MTL ablation이다.

### Lec14-Lec15 - Privacy, DP, secure aggregation

FL은 raw data를 보내지 않아도 gradient/model update가 membership inference나 gradient
inversion에 취약할 수 있다. DP는 noise로 privacy를 보장하지만 utility를 깎고, secure
aggregation은 server가 개별 update를 보지 못하게 하지만 aggregate는 볼 수 있다.

ELSA 적용:

- benchmark 연구에서는 privacy가 1순위는 아니다. 하지만 real robot/institutional demo로
  확장하면 update가 RGB/action trajectory 정보를 누출할 수 있다.
- LoRA-only aggregation은 privacy/communication 측면에서도 좋다.
- DP는 manipulation success에 직접 손해를 줄 가능성이 크므로 마지막 단계다.
- secure aggregation은 Flower/FATE류 framework 관점에서 장기 과제다.

## 4. 첨부 논문별 적용 요약

| 자료 | 푸는 상황 | 핵심 방법 | 우리 적용 |
|---|---|---|---|
| Agnostic FL | train/test client mixture가 다르고 worst-client가 중요 | minimax 또는 client mixture robust objective | mean SR 외에 worst-env/worst-task SR 도입 |
| q-FFL | 평균 성능은 높지만 일부 client가 계속 나쁨 | high-loss client에 `loss^q` 가중 | hard env/camera/task 가중 ablation |
| FedProx | statistical/system heterogeneity로 local drift 발생 | local objective에 proximal term | 이미 구현됨. `mu` grid가 중요 |
| SCAFFOLD | non-IID client drift가 FedAvg를 흔듦 | server/client control variate | FedAvg/FedProx 모두 실패할 때 구현 |
| FedNova | client별 local step 수가 다름 | normalized averaging | VolumeDP-full처럼 runtime 차이 클 때 |
| Power-of-Choice | partial participation에서 어려운 client를 빨리 학습 | candidate 중 high-loss client 선택 | diagnostic sampling, hard-env mining |
| Ditto | global model 하나로 모든 client가 만족하지 않음 | global + local personalized model with prox | local gripper/action head ablation |
| FedExp/FedExP | FedAvg server step size가 보수적/느림 | adaptive server LR/extrapolation | server-only라 먼저 시도 가능 |
| MaxFL | global model이 client requirement를 못 맞춰 참여 유인이 낮음 | threshold 만족 client 수를 최대화 | success threshold 기반 appeal metric |

## 5. ELSA/FLAME 아이디어로 재정의

현재 아이디어를 강의 언어로 쓰면 다음과 같다.

> FLAME manipulation FL의 병목은 단순한 visual non-IID가 아니라
> representation shift, geometry shift, action/controller contract shift,
> temporal gripper event shift가 합쳐진 multi-source heterogeneity다.
> 우리는 frozen DINO/Depth + LoRA + VolumeDP-style spatial tokenization +
> diffusion decoder로 aggregation surface를 줄이고, FedAvg-class optimizer가
> 처리해야 하는 residual heterogeneity를 낮춘다.

이 framing에서 실험 질문은 다음 네 개다.

1. **Aggregation surface**: full model 평균보다 LoRA/projector/decoder만 평균하는 것이
   drift를 줄이는가.
2. **Geometry normalization**: VolumeDP-style voxel/world-frame representation이
   camera/view heterogeneity를 줄이는가.
3. **Action contract**: TGAC transition-weighted BCE/hysteresis/event head가
   gripper timing heterogeneity를 줄이는가.
4. **Optimizer residual**: 위 세 가지를 한 뒤에도 FedAvg가 부족한가, 아니면 FedProx,
   FedExp, SCAFFOLD가 추가 이득을 주는가.

## 6. 구현 repo와 현재 clone 상태

상세 manifest는 `fede/repos_manifest.md`에 따로 저장했다.

| Repo | Path | 바로 볼 부분 |
|---|---|---|
| Flower | `fede/repos/flower` | `baselines/fedprox`, `baselines/fednova`, custom strategy examples |
| FedProx | `fede/repos/FedProx` | `flearn/trainers/fedprox.py`, `run_fedprox.sh` |
| FedNova | `fede/repos/FedNova` | `distoptim/FedNova.py`, `distoptim/FedProx.py` |
| SCAFFOLD | `fede/repos/Scaffold-Federated-Learning` | `server.py`, `client.py`, `ScaffoldOptimizer.py` |
| q-FFL/AFL | `fede/repos/fair_flearn` | `flearn/trainers/qffedavg.py`, `flearn/trainers/afl.py` |
| Ditto | `fede/repos/ditto` | `flearn/trainers_MTL/ditto.py`, personalization baselines |
| FedML | `fede/repos/FedML` | large framework reference, not immediate |
| FATE | `fede/repos/FATE` | secure/industrial FL reference, not immediate |

## 7. ELSA 코드에 붙이는 우선순위

### 7.1 이미 된 것

- FedAvg server aggregation.
- FedProx local proximal loss.
- trainable-only aggregation.
- 2026-05-07 패치로 server `image_channels`, `low_dim_state_dim` config 기반 보강.
- server trainable parameter manifest 저장.

### 7.2 바로 할 만한 것

1. **FedExp/FedExP-style server LR ablation**
   - server가 aggregated delta를 `w <- w + eta_g * delta`로 적용.
   - `eta_g in {0.5, 1.0, 1.5, 2.0}` 또는 bounded adaptive rule부터.
   - SCAFFOLD보다 구현이 작고 current Flower strategy와 잘 맞는다.

2. **q-FFL/MaxFL-style metric weighting**
   - client train loss 또는 held-out env loss/success로 server aggregation weight를 조절.
   - `mean SR`과 `worst-env SR`을 동시에 보고한다.
   - noisy client를 과대반영하지 않도록 minimum samples와 failure hygiene filter 필요.

3. **Power-of-Choice diagnostic selection**
   - random clients와 high-loss clients를 분리해서 실험한다.
   - hard-env mining으로 gripper/camera failure를 빨리 찾되, 최종 성능은 unbiased eval로 보고.

### 7.3 나중에 할 것

1. **SCAFFOLD**
   - client control variate를 저장해야 한다.
   - Flower simulation에서 partition/client identity가 안정적으로 유지되어야 한다.
   - 구현 후에는 extra communication/memory도 보고해야 한다.

2. **FedNova**
   - local steps가 client마다 달라지는 설정부터 만든 뒤 의미가 있다.
   - 현재 동일 local epochs 실험에는 우선순위가 낮다.

3. **Ditto/personalization**
   - global checkpoint evaluation이 main이면 paper 주장은 약하다.
   - camera cohort/local action head/gripper calibration 분석에는 좋다.

## 8. VolumeDP/diffusion decoder 관점의 구체적 적용

### Diffusion decoder

- local update가 action mode/timing을 client별로 당긴다.
- FedProx는 decoder를 과하게 global model 근처에 묶어 multimodal adaptation을 막을 수 있다.
- 기존 pilot에서 FedAvg가 FedProx보다 좋았던 것은 이 설명과 맞다.
- 우선순위:
  1. FedAvg baseline 유지
  2. local_epochs 1/3/5
  3. FedProx small grid
  4. server LR ablation
  5. SCAFFOLD

### VolumeDP

- voxel/world-frame contract가 맞으면 geometry heterogeneity를 줄인다.
- contract가 틀리면 같은 voxel token이 다른 물리 위치를 의미해서 FedAvg가 더 나빠진다.
- aggregate 대상:
  - DINO LoRA
  - depth/visual projector
  - voxel lift/spatial token learner
  - diffusion decoder
- local-only 대상:
  - raw RGB/depth
  - camera K/T
  - per-client voxel tensor
  - frozen DINO/Depth base

### Gripper/TGAC

- 강의 관점에서는 temporal gripper event mismatch가 action distribution heterogeneity다.
- transition-weighted BCE는 high-loss event를 q-FFL처럼 더 보게 하는 local objective
  reweighting이다.
- hysteresis는 training algorithm이 아니라 controller contract 후처리다.
- event head는 personalization과 연결 가능하다. global event head부터 보고,
  camera/task별 local calibration은 후속 ablation으로 둔다.

## 9. 추천 실험 매트릭스

### Stage 0 - 현재 큐/TGAC 정리

- current Ralph fill4 queue 완료 대기.
- success, gripper transition F1, predicted/executed gripper flips, action magnitude 정리.
- TGAC가 non-zero success를 만든 뒤 FL로 이동.

### Stage 1 - FedAvg/FedProx 최소 grid

| Factor | Values |
|---|---|
| model | `DINO LoRA4/8 + diffusion + JV direct` |
| local_epochs | `1, 3, 5` |
| prox_mu | `0, 1e-4, 3e-4, 1e-3, 3e-3` |
| metrics | mean SR, worst-env SR, train loss, gripper transition metrics |

### Stage 2 - Server-side cheap methods

- FedExp/FedExP-style server LR.
- q-FFL/MaxFL-style loss/success weighting.
- Power-of-Choice diagnostic client selection.

### Stage 3 - Heavy methods

- SCAFFOLD only if Stage 1-2 shows genuine client drift.
- FedNova only if local step/runtime heterogeneity is intentionally introduced.
- Ditto only if personalization/camera cohort story becomes central.

## 10. 논문 스토리로 쓰기 좋은 문장

가장 강한 주장은 "우리는 새 FL optimizer를 냈다"가 아니다.

더 좋은 주장은:

> In federated manipulation, optimizer-level drift correction is insufficient
> unless the policy architecture first removes representation, geometry, and
> action-contract heterogeneity. We show that a small aggregation surface built
> from frozen visual priors, LoRA adapters, volumetric tokens, and action-diffusion
> decoders makes simple FedAvg competitive, and then characterize when stronger
> FL objectives are actually needed.

## 11. 참고 링크

- FLAME: https://arxiv.org/abs/2503.01729
- FedAvg: https://arxiv.org/abs/1602.05629
- FedProx: https://arxiv.org/abs/1812.06127
- SCAFFOLD: https://arxiv.org/abs/1910.06378
- FedNova: https://arxiv.org/abs/2007.07481
- q-FFL: https://arxiv.org/abs/1905.10497
- Agnostic FL: https://arxiv.org/abs/1902.00146
- Ditto: https://arxiv.org/abs/2012.04221
- FedExP: https://arxiv.org/abs/2301.09604
- MaxFL: https://arxiv.org/abs/2205.14840
- VolumeDP: https://arxiv.org/abs/2603.17720
- Diffusion Policy: https://arxiv.org/abs/2303.04137
