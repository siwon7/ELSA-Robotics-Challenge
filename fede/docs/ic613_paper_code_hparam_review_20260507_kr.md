# IC613 FL paper/code/hyperparameter review for ELSA

작성일: 2026-05-07

이 문서는 현재 ELSA FL 구현이 강의/PPT, 원 논문, 로컬 클론 reference code와 얼마나 맞는지 점검한 기록이다. 결론부터 쓰면, 지금 커밋 대상 구현은 smoke/ablation용으로 쓸 수 있지만 `SCAFFOLD`, `Power-of-Choice`, `Ditto` 같은 client-state/selection/personalized-model 계열은 아직 구현 범위 밖이다.

## 확인한 로컬 원본

- FedExp paper text: `fede/extracted_text/_2023_ICLR__FedExp.txt`
- FedNova paper text/reference: `fede/extracted_text/_2020_NeurIPS__FedNova.txt`, `fede/repos/FedNova`
- q-FFL/q-FedAvg/AFL reference: `fede/repos/fair_flearn/flearn/trainers/qffedavg.py`, `fede/repos/fair_flearn/flearn/trainers/afl.py`
- MaxFL/MaxGM paper text: `fede/extracted_text/_2024_TMLR__MaxGM_in_FL.txt`
- IC613 lecture algorithm taxonomy: `fede/docs/ic613_all_algorithms_taxonomy_for_elsa_20260507_kr.md`

## 코드 구조 판단

현재 구조는 기존 Flower app을 크게 깨지 않는 방향이다.

- `federated_elsa_robotics/fl_method_registry.py`: method를 `server_strategy`, `client_update`, `local_regularizer`, `parameter_scope`, `client_state` 축으로 분해한다.
- `federated_elsa_robotics/parameter_surfaces.py`: FedPer/FedRep류를 위해 server-aggregated tensor와 client-local tensor를 분리한다.
- `federated_elsa_robotics/server_aggregation.py`: ndarray 기반 순수 aggregation helper다. Flower strategy와 분리되어 smoke test가 쉽다.
- `federated_elsa_robotics/strategies.py`: Flower `FedAvg` 상속부는 유지하고, trainable-only aggregation/checkpointing만 교체한다.
- `client_app.py`와 `server_app.py`: runtime config, manifest, local state path, metrics probe만 연결한다.

이 분리는 괜찮다. 알고리즘을 Flower class마다 새로 파는 것보다, parameter surface와 server aggregation을 별도 모듈로 둔 것이 이후 SCAFFOLD/Ditto 확장에도 낫다.

## 알고리즘별 충실도

### FedAvg/FedProx

- 상태: 사용 가능.
- FedProx는 local objective에 proximal term을 더하고, FedPer류와 함께 쓸 때는 aggregated/shared parameter에만 proximal term을 건다.
- 기본 `prox_mu=1e-3`는 현재 DINOv3 frozen + LoRA + diffusion head smoke에서 과하지 않은 시작점이다. drift가 커지면 `3e-4, 1e-3, 3e-3` grid로 보는 게 낫다.

### FedPer/FedProx-FedPer

- 상태: 사용 가능.
- local-only keywords 기본값은 `policy_fc2`, `diffusion_head`, `multitoken_diffusion_head`, `gripper_head`다.
- 서버 checkpoint는 shared aggregated surface를 저장하고, client-local state는 `<checkpoint-root>/<task>/client_local_state/<run-tag>/partition_<id>_env_<env>.pt`에 저장한다.
- 이 계열은 "server aggregation only"가 아니다. local head가 client에 남으므로 personalized eval script를 같이 써야 한다.

### FedExp

- 상태: paper-form server LR 사용 가능.
- 원 논문 Algorithm 1의 핵심은 `sum_i ||Delta_i||^2 / (2M ||Delta_bar||^2 + eps)`로 server LR을 잡고, 최소 1 이상으로 둔다는 점이다.
- 구현은 ELSA sign convention에 맞춰 `updated - previous` delta를 쓰되 squared norm은 동일하고 최종 update 방향도 맞다.
- smoke 기본값은 `fedexp_min_lr=1.0`, `fedexp_max_lr=3.0`, `server_learning_rate=1.0`으로 맞췄다. 논문은 upper bound가 필수는 아니지만, LoRA/diffusion head 조합에서는 초기 폭주를 막기 위해 clamp를 둔다.

### FedNova

- 상태: vanilla local-SGD normalizer first pass.
- 원 논문/official implementation은 local normalizer, local step 수, momentum/prox variant까지 본다. 현재 구현은 client가 보내는 `local_steps`를 이용해 `delta / tau`를 평균한 뒤 effective tau를 곱한다.
- 모든 client가 같은 local epoch/step을 돌면 FedAvg와 거의 같아야 한다. GPU scheduling, shard size, skipped batch 때문에 step 수가 달라질 때 의미가 생긴다.

### q-FFL / q-FedAvg

- 상태: fair_flearn reference에 맞춘 dynamic-step update가 기본.
- reference code는 `loss^q * grad`와 `h = q * loss^(q-1) * ||grad||^2 + loss^q / lr` denominator를 쓴다.
- ELSA 구현은 delta sign만 변환해 같은 방향으로 update한다. `qffl_learning_rate`는 반드시 local optimizer `model.learning_rate`와 맞춰야 하므로 기본을 둘 다 `0.0003`으로 둔다.
- `qffl_max_delta_multiplier=2.0`은 diffusion/LoRA smoke 안정성용 clipping이다. 순수 논문식만 보려면 값을 0으로 두면 된다.
- `--no-qffl-dynamic-step`은 논문식 q-FedAvg가 아니라 loss-weighted ablation이다.

### AFL

- 상태: sampled-client AFL approximation.
- fair_flearn AFL은 전체 client loss에 대한 lambda simplex update가 핵심이다. Flower partial participation에서는 매 round 관측 client만 loss를 보므로, 현재 구현은 observed client/domain lambda를 누적하고 simplex projection한다.
- privacy/secure aggregation story와는 충돌한다. client별 loss가 서버에 공개되기 때문이다.

### MaxFL / MaxGM

- 상태: MaxFL-inspired threshold weighting first pass.
- 원 논문은 client별 requirement/benefit threshold를 두고 `sigma(F_i(w)-rho_i)`류 objective를 최적화한다.
- 현재 구현은 per-client `rho_i`가 없으므로 loss threshold 또는 round mean threshold 주변 sigmoid-derivative weight를 쓴다. 즉 "hard env를 더 보게 하는 MaxFL 계열 ablation"이지, full MaxFL 프로토콜은 아니다.
- full MaxFL로 올리려면 env별 success/loss requirement, local-only model benefit, participation gain metric을 client telemetry로 추가해야 한다.

## 현재 우선 실행 순서

1. 기존 active queue는 건드리지 않는다.
2. CPU smoke: `fedavg`, `fedprox`, `fedper_head`, `fedexp`, `qfedavg`를 1-2 round로 import/runtime 확인한다.
3. 실제 GPU ablation은 `fedavg/fedprox` baseline 이후 `fedper_head -> fedprox_fedper_head -> fedexp -> qfedavg` 순서가 안전하다.
4. `fednova`는 local step heterogeneity가 관측될 때만 우선순위를 올린다.
5. `afl/maxfl`은 worst-env metric과 gripper-event metric이 안정화된 뒤 돌린다.

## 남은 구현 후보

- SCAFFOLD: server/client control variate state가 필요하다.
- Power-of-Choice: client sampling protocol을 바꿔야 한다.
- Ditto/pFedMe류: full personalized model state와 별도 regularized local objective가 필요하다.
- secure aggregation compatibility: q-FedAvg/AFL/MaxFL처럼 per-client loss가 필요한 전략과 충돌하므로 별도 privacy mode가 필요하다.
