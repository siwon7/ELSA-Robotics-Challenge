# IC613 Federated Learning 심화 독해 및 ELSA 적용 계획

작성일: 2026-05-07
대상: `ELSA-Robotics-Challenge`의 Flower 기반 federated 학습, FLAME/VolumeDP/diffusion decoder/TGAC 계열 실험

## 0. 이번 패스에서 실제로 확인한 것

사용자가 준 `IC613 lec01 Intro.7z`는 PPT가 아니라 PDF 묶음이었다. `fede/`에 harness `63-research-assistant`를 적용했고, `bsdtar`로 강의/논문 PDF를 풀었다.

- 원본 PDF: `fede/course_materials/ic613_lectures/`
- 추출 텍스트: `fede/extracted_text/*.txt`
- 인벤토리: `fede/extracted_text/pdf_inventory.tsv`
- 페이지 단위 색인: `fede/extracted_text/page_index.tsv`
- 키워드 색인: `fede/extracted_text/keyword_hits.tsv`
- 레포 클론: `fede/repos/`
- 모듈화 초안: `fede/modules/`

총 24개 PDF를 확인했다. 강의 15개는 슬라이드라 텍스트가 짧은 페이지가 많고, 첨부 논문 9개는 본문까지 추출된다. OCR 엔진은 없어서 그림 안에만 있는 수식/표는 완전 추출되지 않는다. 대신 강의 목차, 슬라이드 제목, 논문 본문, 클론한 레포 구현을 같이 대조했다.

## 1. 먼저 잡아야 할 연구 관점

지금 ELSA 쪽 문제는 단순히 "vision backbone이 약하다"가 아니다. Ralph/VolumeDP/diffusion 계열에서 더 직접적인 병목은 다음 세 가지로 보인다.

1. Temporal gripper event grounding: gripper transition이 sparse event라 평균 MSE나 일반 BCE로는 묻힌다.
2. Action/controller contract: action chunk, gripper bit, joint/ee servo 계약이 어긋나면 좋은 representation도 live rollout에서 깨진다.
3. Client/env heterogeneity: env shard, task shard, robot/controller 차이가 FedAvg 평균에서 drift와 worst-env failure를 만든다.

따라서 FL 알고리즘은 1번과 2번을 해결하는 대체재가 아니라, 그 다음 단계의 optimizer/aggregation layer다. 지금 우선순위는 gripper transition-weighted BCE, hysteresis, event head로 local objective를 안정화한 뒤, FedExp/q-FedAvg/MaxFL/FedNova/SCAFFOLD를 순서대로 붙이는 것이다.

## 2. 현재 ELSA FL 코드의 실제 계약

이번 패스 전 서버/클라이언트 흐름은 다음과 같았다.

- `federated_elsa_robotics/server_app.py`
  - 서버가 config에서 `image_channels`, `low_dim_state_dim`, `action_dim`을 추론해 `Agent`를 만든다.
  - `get_trainable_parameter_manifest(agent, config)`로 aggregated/local parameter surface를 기록한다.
  - `TrainableOnlyFederatedStrategy`가 `fedavg`, `fedexp`, `fednova`, `qfedavg`, `afl`, `maxfl` server aggregation 결과를 aggregated parameter에 set하고 checkpoint를 저장한다.
- `federated_elsa_robotics/client_app.py`
  - partition id를 env shard로 매핑한다.
  - 클라이언트 sample에서 실제 image/state/action shape를 확인한다.
  - `train()` 후 `get_weights()`와 `{"train_loss": train_loss}` 중심으로 서버에 보냈다.
- `federated_elsa_robotics/task.py`
  - `iter_trainable_parameters()` 기준으로 trainable tensor만 FL aggregation 대상이다.
  - `prox_mu > 0`이면 local objective에 FedProx proximal term을 더한다.

기존 계약으로 가능했던 것:

- FedAvg
- FedProx local proximal term
- trainable surface logging
- config/shape mismatch 방지

기존 계약으로 부족했던 것:

- FedExp/q-FedAvg/FedNova/SCAFFOLD에 필요한 delta norm, local step 수, 시작점 loss, client id별 state가 없다.
- server aggregation은 아직 Flower `FedAvg.aggregate_fit()`에 많이 의존한다.
- metrics aggregation은 전체 `train_loss` 평균뿐이라 env/task/worst-client behavior를 볼 수 없다.

이 패스에서 aggregation rule은 유지하되 client metrics 계약은 확장했다. 이제 기본적으로 `delta_norm_sq`, `local_steps`, `num_batches`, `partition_id`, `env_id`, `task`, `trainable_manifest_hash`가 fit metrics에 들어가며, `metrics-probe-batches > 0`일 때 `pre_train_loss`와 `post_train_loss`도 기록된다.

## 3. 강의별 독해와 ELSA 적용

### Lecture 01 Intro

핵심 상황: 왜 중앙집중 ML에서 distributed/federated ML로 넘어가는가를 잡는다. 데이터센터 data/model parallel과 edge FL을 대비하고, 시스템/네트워크/최적화가 같이 들어가야 한다고 설명한다.

ELSA 적용: 우리 상황은 엄밀한 스마트폰 cross-device는 아니지만, env/task/robot shard를 client로 보는 cross-silo 성격이 강하다. 원데이터 privacy보다 heterogeneity와 controller contract가 핵심이다.

구현 의미: FL을 붙인다는 말은 단순 평균 스크립트를 만든다는 뜻이 아니라, client runtime, local compute budget, shard별 data distribution, aggregation state를 명시해야 한다.

### Lecture 02 SGD and Variants

핵심 상황: GD/SGD/mini-batch, learning rate, momentum, AdaGrad/Adam의 기본 역할을 다룬다.

ELSA 적용: 현재 local optimizer가 Adam이다. FedProx term이나 diffusion decoder의 action loss를 바꾸면 local optimizer가 실제로 푸는 objective가 바뀐다. FL 성능 해석 전에 local optimizer 조건을 고정해야 한다.

구현 의미: FL ablation은 `local_epochs`, `batch_size`, `learning_rate`, `weight_decay`, `prox_mu`를 함께 기록해야 한다. client update norm도 optimizer별로 달라지므로 FedExp/q-FedAvg에 필요하다.

### Lecture 03 SGD Convergence Analysis

핵심 상황: Lipschitz smoothness, strong convexity, stochastic gradient variance, error floor를 배운다.

ELSA 적용: robotics imitation은 비볼록이고 stochastic하다. 따라서 "수렴 보장"보다 gradient/update variance와 drift 진단 지표를 실험 로그에 넣는 게 실용적이다.

구현 의미: round별 평균 loss 하나로는 부족하다. client delta norm, delta cosine similarity, local loss before/after, worst-client metric을 기록해야 한다.

### Lecture 04 SGD Convergence Analysis 2

핵심 상황: non-convex SGD, SAG/SAGA/SVRG 같은 variance reduction을 다룬다.

ELSA 적용: SCAFFOLD/FedVARP는 이 variance reduction 관점의 federated 버전이다. gripper event가 sparse해서 gradient variance가 큰 경우에도 "어느 client가 어떤 방향으로 계속 튀는지"를 줄이는 관점이 맞다.

구현 의미: SCAFFOLD를 하려면 client별 control variate가 있어야 한다. Flower simulation에서 client identity가 안정적으로 유지되는지 먼저 보장해야 한다.

### Lecture 05 Federated Learning Intro

핵심 상황: FL 정의, raw data는 local에 두고 update만 교환하는 구조, FedAvg, cross-device/cross-silo, participation fraction/local epoch/batch size 효과를 설명한다.

ELSA 적용: 우리의 client는 env shard 또는 robot/site shard다. privacy보다 shard별 dynamics와 task balance가 중요하다.

구현 의미: `fraction_fit`, `local_epochs`, `num_partitions`를 바꿀 때 같은 task/env split에서만 비교해야 한다. trainable manifest가 다르면 FedAvg 결과 비교가 무효다.

### Lecture 06 Heterogeneity 1

핵심 상황: data heterogeneity가 FedAvg drift를 만든다. FedProx는 proximal term으로 client model discrepancy를 줄이고, SCAFFOLD는 control variate로 drift를 보정한다. FedExp는 server LR로 local correction slowdown을 줄인다.

ELSA 적용: env별 camera pose, object pose, gripper timing, controller response가 local objective 차이를 만든다. FedProx는 이미 코드에 들어갔으므로 가장 낮은 위험의 baseline이다.

구현 의미: FedProx sweep은 이미 가능하다. 그 다음은 FedExp-style server LR이다. SCAFFOLD는 stateful client control이 필요하므로 바로 넣지 않는다.

### Lecture 07 Heterogeneity 2

핵심 상황: computational heterogeneity와 objective inconsistency를 다룬다. local step 수가 client마다 다르면 FedAvg가 원래 global objective가 아닌 다른 objective로 갈 수 있다. FedNova가 이를 normalized averaging으로 해결한다.

ELSA 적용: diffusion decoder나 VolumeDP full model은 client별 runtime 차이가 크고, data shard 크기도 다르다. 일부 client가 더 많은 local steps를 수행하면 그 client update가 과도하게 반영될 수 있다.

구현 의미: FedNova는 client가 `local_steps`, normalized delta 또는 local normalizer를 metric/update와 함께 보내야 한다. 현재의 final weights만 평균하는 계약으로는 논문식 FedNova가 아니다.

### Lecture 08 Heterogeneity 3 and Partial Selection

핵심 상황: FedNova를 재정리하고, partial participation variance reduction(FedVARP)와 biased client selection을 소개한다.

ELSA 적용: active_workers=4/16 같은 제한이 있을 때 partial participation 자체가 variance를 만든다. 이때 high-loss env를 계속 뽑으면 빨라질 수 있지만 solution bias가 생긴다.

구현 의미: Power-of-Choice는 production 기본값이 아니라 diagnostic selection으로 시작해야 한다. random-client eval을 같이 두지 않으면 성능 개선처럼 보이는 bias를 만들 수 있다.

### Lecture 09 Partial Selection

핵심 상황: FedAvg partial participation의 convergence, SAG/SAGA 관점의 FedVARP, Power-of-Choice 및 adaptive d를 더 자세히 다룬다.

ELSA 적용: 현재 16개 env 중 일부만 돌리는 scheduling이면, 이전 round에서 빠진 env의 stale gradient/metric이 생긴다.

구현 의미: FedVARP/SAGA식 memory를 넣으려면 client별 과거 delta 저장이 필요하다. 지금은 우선 per-env metric cache를 만들고, 나중에 selection만 조절하는 쪽이 안전하다.

### Lecture 10 Fairness and Participation Incentives

핵심 상황: biased selection, fairness in FL, AFL, q-FFL, MaxFL/participation incentives를 다룬다.

ELSA 적용: 평균 rollout 성공률이 좋아도 특정 env/task가 계속 실패하면 Ralph benchmark에는 치명적이다. q-FFL/AFL/MaxFL은 "평균"보다 "worst env/client"를 끌어올리는 도구다.

구현 의미: q-FedAvg는 단순 `loss^q` 가중이 아니다. 시작점 loss, pseudo-gradient norm, denominator가 필요하다. MaxFL은 client별 requirement threshold가 필요하다.

### Lecture 11 Participation Incentives and Personalization

핵심 상황: MaxFL을 복습하고 global-only/local-only/fine-tuning/Ditto/clustering personalization을 다룬다.

ELSA 적용: site/robot별 로컬 adaptation이 허용된다면 Ditto가 강하다. 하지만 최종 제출이 하나의 global checkpoint만 받는다면 personalization 결과는 메인 점수에 바로 쓰기 어렵다.

구현 의미: Ditto는 global model과 local personalized model을 동시에 유지한다. 현재 checkpoint/eval 스크립트는 global-only라 별도 저장/평가 루프가 필요하다.

### Lecture 12 Personalization 2

핵심 상황: Ditto, clustering, meta-learning(Per-FedAvg), multi-task learning, shared trunk, MMoE, MGDA를 다룬다.

ELSA 적용: VolumeDP에서는 global trunk(vision/geometry)와 local decoder/head(컨트롤러/로봇별 action)를 분리할 수 있다. 하지만 local head를 쓰면 unseen env 일반화와 제출 조건을 다시 봐야 한다.

구현 의미: FedPer/FedRep 스타일로 trainable surface를 나누는 설계가 가능하다. 현재 `iter_trainable_parameters()`가 단일 trainable set이므로 layer group manifest가 필요하다.

### Lecture 13 Personalization 3

핵심 상황: MTL, model merging, task arithmetic까지 확장한다.

ELSA 적용: task arithmetic은 task/env별 delta를 분석해 "gripper event delta", "joint servo delta", "camera pose delta"처럼 분리할 수 있는 가능성을 준다. 그러나 지금은 검증 장치가 없으므로 연구 보조 분석으로만 둔다.

구현 의미: task delta를 저장하려면 round별 client delta checkpoint가 필요하다. 이건 저장 공간과 privacy surface가 커지므로 우선 norm/cosine metric부터 시작한다.

### Lecture 14 Privacy 1

핵심 상황: gradient inversion, membership inference, differential privacy, secure aggregation을 다룬다.

ELSA 적용: vision+proprio robotics gradient는 raw image/trajectory 정보를 누출할 수 있다. 실제 institution/robot silo FLAME 환경이라면 privacy는 부가가 아니라 핵심 조건이 된다.

구현 의미: 현재 연구 harness에서는 privacy보다 성능/재현성이 먼저다. 하지만 논문 스토리에는 secure aggregation 가능성과 DP 비용을 분리해서 써야 한다. local DP는 action precision을 크게 해칠 수 있어 마지막 단계다.

### Lecture 15 Privacy 2

핵심 상황: DP와 secure aggregation을 비교하고 one-time pad, dropout, Shamir secret sharing을 설명한다.

ELSA 적용: secure aggregation은 서버가 개별 client update를 보지 못하게 하며 성능 손실이 작다. DP는 noise로 성능 손실이 생긴다.

구현 의미: q-FedAvg/MaxFL처럼 개별 loss/weight를 서버가 보는 알고리즘은 secure aggregation과 충돌할 수 있다. privacy mode를 목표로 하면 per-client metric을 줄이거나 암호화된 aggregate-friendly 전략을 써야 한다.

## 4. 첨부 논문별 구현 계약

### FedProx

문제: statistical/system heterogeneity가 있는 FedAvg에서 client local update가 각자 local optimum 쪽으로 drift한다.

알고리즘: local objective에 `mu/2 * ||w - w_global||^2`를 더해 local model이 global 시작점에서 너무 멀어지지 않게 한다.

레포 확인:

- `fede/repos/FedProx/flearn/trainers/fedprox.py`
- `fede/repos/FedProx/flearn/optimizer/pgd.py`

구현 포인트:

- 서버 aggregation은 FedAvg와 거의 같다.
- client local train loop만 바꾸면 된다.
- 공식 구현은 straggler/drop_percent와 dissimilarity logging을 같이 둔다.

ELSA 상태:

- `federated_elsa_robotics/task.py`에 proximal term이 이미 있다.
- `prox_mu` config/override가 있으므로 바로 sweep 가능하다.

주의:

- FedProx는 objective inconsistency를 완전히 없애지 않는다.
- `mu`가 크면 local adaptation이 둔해지고 gripper event learning이 느려질 수 있다.

### SCAFFOLD

문제: non-IID client에서 FedAvg local updates가 client-drift를 만든다.

알고리즘: server control variate `c`와 client control variate `c_i`를 유지하고, local gradient를 `g_i(y) + c - c_i`로 보정한다.

레포 확인:

- `fede/repos/Scaffold-Federated-Learning/ScaffoldOptimizer.py`
- `fede/repos/Scaffold-Federated-Learning/server.py`
- `fede/repos/flower/baselines/niid_bench/niid_bench/server_scaffold.py`
- `fede/repos/flower/baselines/niid_bench/niid_bench/client_scaffold.py`

구현 포인트:

- client는 stateful해야 한다.
- stable `client_id -> c_i` 매핑이 필요하다.
- client는 model delta뿐 아니라 control delta도 보낸다.
- trainable parameter surface가 바뀌면 control variate shape도 무효다.

ELSA 적용:

- env별 지속적인 drift가 확인된 뒤 넣는다.
- VolumeDP full처럼 큰 trainable surface에서는 control variate 메모리가 부담이다.
- LoRA/event-head 같은 작은 trainable surface에서 먼저 시험한다.

### FedNova

문제: client별 local update 수가 다르면 FedAvg가 원래 global objective가 아닌 mismatched objective로 갈 수 있다.

알고리즘: 각 client의 누적 local change를 local normalizer로 나눈 normalized gradient로 만들고, effective local steps로 server update를 구성한다.

레포 확인:

- `fede/repos/FedNova/distoptim/FedNova.py`
- `fede/repos/flower/baselines/fednova/fednova/strategy.py`

Flower baseline 구현에서 서버는 client metrics의 `tau`, `local_norm`, `weight`를 사용한다. 즉 final weights만 평균하면 FedNova가 아니다.

ELSA 적용:

- diffusion decoder나 VolumeDP full에서 client runtime/step 수가 달라질 때 중요하다.
- 현재처럼 모든 client가 같은 `local_epochs`와 같은 DataLoader 길이로 돌면 우선순위가 낮다.
- active queue에서 slow/fast client 때문에 실제 local update 수가 달라지는지부터 기록해야 한다.

필요 코드:

- client: `local_steps`, `num_batches`, `optimizer_normalizer`, `delta_norm_sq` metric 반환
- server: returned weights를 delta로 바꾼 뒤 FedNova formula로 aggregate

### FedExp

문제: heterogeneity를 줄이려고 client LR을 작게 하면 global convergence가 느려진다.

알고리즘: client update를 pseudo-gradient로 보고 server learning rate를 adaptive하게 키운다.

레포 확인:

- Flower의 FedAvgM baseline이 pseudo-gradient/server momentum 구현 구조 참고점이다.
- `fede/repos/flower/baselines/fedavgm/`

ELSA 적용:

- 가장 먼저 production화할 만하다.
- client state가 필요 없고 server aggregation만 바꾸면 된다.
- trainable-only parameter surface와 잘 맞는다.

필요 코드:

- server가 현재 global weights와 aggregated weights 차이를 delta로 계산
- `server_lr`를 bounded grid 또는 FedExP-inspired heuristic으로 적용
- `server_lr in {1.0, 1.5, 2.0, adaptive}` ablation

### Power-of-Choice

문제: random partial participation은 느리고, high-loss client를 더 자주 뽑으면 수렴이 빨라질 수 있다.

알고리즘: 후보 `d`개 client의 local loss를 보고 높은 loss client를 선택한다.

레포/논문 확인:

- 첨부 `[2021 ICLR] power-of-choice.pdf`
- 구현 레포는 별도 official code가 명확하지 않아 Flower custom selection으로 구현하는 편이 낫다.

ELSA 적용:

- gripper transition failure가 높은 env를 더 자주 학습시키는 diagnostic으로 유용하다.
- 단, selection bias 때문에 random eval을 같이 유지해야 한다.

필요 코드:

- stale client loss cache
- 후보군 sampling 후 high-loss top-k
- random baseline과 같은 wall-clock budget 비교

### AFL

문제: training distribution과 test/client mixture가 다를 수 있고, 평균 objective가 특정 client를 희생할 수 있다.

알고리즘: client mixture weight `lambda`를 simplex 위에서 올려 worst mixture에 강한 global model을 학습한다.

레포 확인:

- `fede/repos/fair_flearn/flearn/trainers/afl.py`
- 구현은 `latest_lambdas`를 loss 방향으로 업데이트하고 simplex projection을 한다.

ELSA 적용:

- 16개 env 정도의 small cross-silo에서는 실험 가능하다.
- 많은 client에서는 lambda state가 커지고 noisy해진다.

필요 코드:

- server-side `lambda_by_client`
- client별 loss metric
- projected lambda update
- aggregation weight를 `lambda * num_examples` 또는 lambda 중심으로 재정의

### q-FFL / q-FedAvg

문제: 평균 성능은 유지하면서 low-performing client의 성능 분산을 줄인다.

알고리즘: q-FFL objective는 high-loss client에 더 큰 weight를 준다. q-FedAvg 구현은 단순 `loss^q` 평균이 아니라, client 시작점 loss와 pseudo-gradient norm으로 denominator `h_k`를 계산한다.

레포 확인:

- `fede/repos/fair_flearn/flearn/trainers/qffedavg.py`
- `fede/repos/fair_flearn/flearn/utils/tf_utils.py`

원 구현의 핵심:

- `loss = c.get_loss()`는 local training 전 global model에서의 loss다.
- `grads = (weights_before - new_weights) / learning_rate`
- `Delta_k = loss^q * grads`
- `h_k = q * loss^(q-1) * ||grads||^2 + (1/lr) * loss^q`
- server는 `sum(Delta_k) / sum(h_k)` 형태로 update한다.

ELSA 적용:

- worst-env gripper miss를 줄이는 데 적합하다.
- metric loss를 어떤 loss로 할지 중요하다. 전체 action MSE보다 transition-weighted gripper loss 또는 rollout failure proxy가 더 맞을 수 있다.

필요 코드:

- client가 local train 전 loss를 측정
- returned weights와 starting weights로 pseudo-gradient norm 계산
- server가 q-FedAvg denominator까지 반영

### MaxFL / MaxGM

문제: client가 global model을 쓸 이유가 없으면 참여 incentive가 낮다. 각 client requirement를 만족하는 global model appeal을 최대화한다.

알고리즘: client requirement threshold `rho_k`를 두고, `F_k(w) - rho_k` 근처의 client update를 더 중요하게 본다. sigmoid relaxation과 adaptive server LR를 사용한다.

레포 상태:

- confirmed official git은 못 찾았다. 첨부 논문과 수식 기반으로 직접 구현하는 쪽이 현실적이다.

ELSA 적용:

- requirement를 "local-only baseline보다 나은 validation RMSE", "gripper transition F1 threshold", "minimum rollout success"로 둘 수 있다.
- unseen env까지 global model appeal을 올리는 스토리에 잘 맞는다.

필요 코드:

- `rho_by_client` 저장
- warm-up local-only 또는 baseline round로 threshold 측정
- appeal weight와 adaptive server LR 적용

### Ditto

문제: 하나의 global model이 fairness와 robustness를 동시에 만족하기 어렵다. personalization으로 client별 model을 global anchor에 묶어 둔다.

알고리즘: global model `w`를 학습하면서 client별 personalized model `v_k`를 `F_k(v_k) + lambda/2 ||v_k - w||^2`로 업데이트한다.

레포 확인:

- `fede/repos/ditto/flearn/trainers/ditto.py`
- local model dict와 global model을 함께 유지한다.

ELSA 적용:

- FLAME에서 robot/site별 decoder adapter를 유지할 수 있으면 강력하다.
- challenge 제출이 global checkpoint 하나라면 메인 결과보다는 analysis/side experiment로 둔다.

필요 코드:

- local personalized checkpoint 저장
- global eval과 personalized eval을 분리
- local adaptation budget 명시

## 5. VolumeDP / diffusion decoder / FLAME 환경에서의 선택

### Diffusion decoder가 있을 때

diffusion decoder는 sampling step, action chunk, noise schedule 때문에 local update norm 변동이 커질 수 있다. 이 경우 다음 순서가 안전하다.

1. trainable surface를 줄인다: full model보다 event head, action head, LoRA/adapters부터 FL.
2. FedExp로 server LR만 조절한다.
3. q-FedAvg/MaxFL로 worst-env를 올린다.
4. 실제 local step 수가 client마다 달라지면 FedNova를 넣는다.
5. 반복적으로 drift가 남으면 SCAFFOLD를 넣는다.

### VolumeDP full 구조일 때

VolumeDP full은 geometry/vision trunk, temporal/action decoder, gripper/event head의 역할이 다르다.

- global로 묶기 좋은 것: geometry prior, frozen/backbone adapter, generic temporal encoder
- local로 남길 수 있는 것: robot/controller-specific action head, calibration-sensitive decoder
- fairness 대상으로 볼 것: env별 event miss, task별 success, trajectory terminal failure

제출이 global-only면 local head personalization은 메인 실험이 아니다. 하지만 analysis로 "global trunk + local small head"를 보여주면 FLAME/real multi-robot 확장성 주장이 좋아진다.

### Privacy/secure aggregation까지 포함할 때

- Secure aggregation은 성능 손실 없이 privacy surface를 줄이는 방향이다.
- DP는 update noise가 action precision과 gripper event를 망칠 수 있어 late-stage ablation이다.
- q-FedAvg/MaxFL/AFL은 per-client loss/weight를 서버가 봐야 하므로 privacy mode와 충돌한다.
- privacy story가 필요하면 먼저 FedAvg/FedProx/FedExp + secure aggregation compatibility를 논의하고, fairness metric은 aggregate-only로 제한하는 방향이 안전하다.

## 6. 구현 로드맵

### Phase A: 관측/로그 계약 보강

목표: FL 알고리즘을 붙이기 전에 client update를 해석할 수 있게 만든다.

필요 metric:

- `client_id`, `partition_id`, `env_id`, `task`
- `num_examples`, `num_batches`, `local_epochs`, `local_steps`
- `pre_train_loss`, `post_train_loss`
- `delta_norm_sq`
- `trainable_manifest_hash`
- optional: `gripper_transition_loss`, `gripper_transition_f1`, `event_head_loss`

리스크: pre-train loss를 full train set으로 측정하면 비용이 커진다. 작은 probe loader 또는 first N batches로 시작한다.

### Phase B: FedExp-style server LR

목표: current FedAvg를 거의 유지하면서 server update만 조절한다.

구현:

- strategy에서 current global weights와 aggregated FedAvg weights의 delta를 계산한다.
- bounded `server_lr`를 적용한다.
- `server_lr=1.0`이 FedAvg baseline이다.

추천 sweep:

- `server_lr_fixed`: 1.0, 1.5, 2.0
- `server_lr_adaptive`: norm ratio bounded to [1.0, 5.0]

### Phase C: q-FedAvg/MaxFL metric-aware aggregation

목표: average loss가 아니라 worst/env failure를 올린다.

q-FedAvg:

- client 시작점 loss를 metric으로 반환한다.
- server가 pseudo-gradient/delta norm으로 `h_k`를 만든다.
- q는 `0, 0.5, 1, 2` 정도부터 시작한다.

MaxFL:

- `rho_k`를 local-only baseline 또는 warm-up global baseline으로 정한다.
- `F_k(w) - rho_k` 근처 client를 강조한다.
- threshold를 너무 aggressive하게 잡으면 아무 client도 appeal 근처에 없어서 update가 불안정하다.

### Phase D: FedNova

목표: heterogeneous local update 수가 실제로 있는 경우 objective inconsistency를 줄인다.

조건:

- client별 `local_steps`가 다르거나 straggler partial update를 허용해야 의미가 있다.
- 모든 client가 같은 epoch/loader step이면 먼저 할 필요가 낮다.

구현:

- client가 local normalizer와 tau를 반환한다.
- server는 final weights 평균이 아니라 normalized delta aggregate를 한다.

### Phase E: SCAFFOLD

목표: persistent client drift를 control variate로 줄인다.

조건:

- stable client identity
- trainable parameter shape 고정
- server/client state checkpoint

첫 실험:

- full VolumeDP가 아니라 LoRA/event-head trainable surface
- client 수 4 또는 8
- short rounds로 shape/state correctness만 먼저 검증

### Phase F: Ditto/FedPer/FedRep personalization

목표: FLAME/robot/site별 local adaptation 가능성을 보인다.

주의:

- global-only benchmark와 personalized benchmark를 섞으면 안 된다.
- local model checkpoint와 eval protocol을 별도로 둔다.

## 7. 이미 만든 모듈과 의미

`fede/modules/strategy_blueprints.py`는 production strategy가 아니라 구현 전 수식/상태 sanity check용 dependency-light helper다.

들어간 것:

- FedProx mu grid
- Power-of-Choice candidate selection
- AFL simplex projection/lambda update
- q-FedAvg numerator/denominator helper
- MaxFL appeal weight/server LR helper
- FedNova effective tau helper
- FedExp paper-form bounded server LR helper
- nested delta norm smoke helper

이 파일을 바로 training path에 import하지 않았기 때문에 현재 Ralph queue에는 영향이 없다.

추가로 active aggregation semantics를 바꾸지 않는 telemetry 기반을 main FL 코드에 넣었다.

- `federated_elsa_robotics/task.py`: trainable manifest hash, optional pre/post train loss probe, local step/batch count 반환.
- `federated_elsa_robotics/client_app.py`: returned update의 `delta_norm_sq`, partition/env/task, local steps, manifest hash를 Flower fit metrics로 반환.
- `federated_elsa_robotics/server_app.py`: `train_loss` 외에 delta/local-step/pre-post loss 평균과 max delta를 aggregate.
- `pyproject.toml`: `metrics-probe-batches = 0` 기본값. 0이면 추가 pre/post probe pass를 하지 않는다.

따라서 기본 FedAvg 결과는 유지하면서, 나중에 `metrics-probe-batches`를 1 이상으로 켜면 q-FedAvg/MaxFL에 필요한 시작점 loss를 기록할 수 있다.

후속으로 server aggregation만 바꾸는 구조의 한계를 줄이기 위해 `fedper_head`와 `fedprox_fedper_head`도 세팅했다. 이 preset들은 shared trainable body만 서버가 aggregate하고, diffusion/action/gripper head는 client-local state로 저장한다. 자세한 축 분해는 `fede/docs/fl_method_axes_and_non_server_only_modularization_20260507_kr.md`에 정리했다.

## 8. 클론한 레포별 사용 판단

- `fede/repos/flower`: ELSA가 이미 Flower 기반이라 가장 실용적인 reference. FedNova/FedAvgM/FedBN/FedRep/FedPer baseline 구조를 참고한다.
- `fede/repos/FedProx`: FedProx local objective와 straggler simulation reference.
- `fede/repos/FedNova`: normalized averaging의 공식 구현 reference.
- `fede/repos/Scaffold-Federated-Learning`: SCAFFOLD control variate 구현 reference. 공식성은 약하지만 PyTorch 구조가 읽기 쉽다.
- `fede/repos/fair_flearn`: AFL/q-FFL 공식 계열 구현 reference.
- `fede/repos/ditto`: Ditto personalization 공식 계열 구현 reference.
- `fede/repos/FedML`: production framework reference. ELSA에 vendor하지 말고 examples/algorithms를 참고한다.
- `fede/repos/FATE`: secure aggregation/industrial FL architecture reference. 현재 실험 path에는 무겁다.

## 9. 다음 실험 우선순위

현재 살아 있는 Ralph queue는 건드리지 않는다. 다음 실험 준비 순서는 다음이 맞다.

1. TGAC local objective 확정: transition-weighted BCE, hysteresis, event head.
2. client telemetry 추가: local steps, pre/post loss, delta norm, env/task metric.
3. FedExp server LR: 가장 작고 안전한 aggregation change.
4. q-FedAvg/MaxFL: worst-env 또는 gripper-event failure 개선 목적.
5. FedNova: local step heterogeneity가 실제로 관측되면.
6. SCAFFOLD: drift metric이 지속적으로 크면.
7. Ditto/FedPer: global-only 결과와 분리한 personalization story로.

핵심은 FL 알고리즘 이름을 많이 붙이는 것이 아니라, 각 알고리즘이 요구하는 client/server state를 실험 로그와 코드 계약에 정확히 반영하는 것이다.
