# IC613 PPT 전체 알고리즘 Taxonomy와 ELSA 적용안

작성일: 2026-05-07
대상 자료: `IC613 lec01 Intro.7z`에서 추출한 lecture 1-15 PDF 텍스트와 함께 제공된 핵심 논문 PDF 텍스트.
산출물: 전체 분류표 `fede/_workspace/ic613_algorithm_taxonomy.tsv`

## 핵심 결론

지금 ELSA에 가장 중요한 구분은 알고리즘 이름이 아니라 **상태와 의사결정이 어디에 남는가**이다.

- **Flower/FedAvg처럼 server aggregation만 하는 축**: client는 매 round local train 후 weight/update만 보내고, 서버가 평균 또는 server learning-rate update를 한다. client별 persistent state는 없다. FedAvg, FedSGD, FedAvg+server LR, FedExp가 여기에 가깝다.
- **서버 aggregation이지만 client telemetry가 필요한 축**: 서버가 여전히 최종 aggregation을 하지만, client loss, delta norm, local steps, benefit/appeal 같은 metric 없이는 제대로 구현할 수 없다. FedNova, q-FedAvg/q-FFL, AFL, MaxFL이 여기에 들어간다.
- **client local objective를 바꾸는 축**: 서버는 평균을 하더라도 client loss 자체가 달라진다. FedProx가 대표적이고 SCAFFOLD는 control variate까지 들어간다.
- **client-local state/local aggregation이 있는 축**: client별 head, personalized model, cluster model, local adaptation state가 계속 남는다. Ditto, FedPer류, clustering, Per-FedAvg, MTL/shared-trunk-local-head, MMoE, MGDA, model merging/task arithmetic이 여기에 해당한다.
- **privacy/security 축**: DP와 secure aggregation은 성능 개선 알고리즘이 아니라 protocol/보안 축이다. 현재 Ralph/ELSA benchmark에서 먼저 넣으면 신호가 흐려질 가능성이 높다.

우리 쪽 연구 가설로 번역하면, 병목을 단순 vision backbone이 아니라 **temporal gripper event grounding + action/controller contract**로 보고 있기 때문에, 서버-only FedAvg/FedProx만으로 끝내면 부족하다. VolumeDP/DINO/FLAME류 shared representation은 global로 묶고, diffusion/action/gripper head는 client-local 또는 personalized로 유지하는 축을 반드시 실험해야 한다.

## 분류 기준

| 축 | 의미 | 서버만 바꾸면 되는가 | client code/state 필요성 |
| --- | --- | --- | --- |
| Server aggregation only | 서버가 weight/update를 평균하거나 server LR만 조절 | 대체로 가능 | persistent client state 없음 |
| Server aggregation + telemetry | 서버가 client metric을 보고 update weight/step/selection을 바꿈 | strategy는 서버 중심 | client가 loss, delta norm, local steps 등을 반환해야 함 |
| Client local objective/correction | client가 local objective나 update rule을 바꿈 | 불가능 | train loop 수정 필요 |
| Client-local persistent state | client별 parameter/model/state가 round를 넘어 유지됨 | 불가능 | local checkpoint/load/eval path 필요 |
| Client selection/participation | 어떤 client를 학습시킬지 바꿈 | Flower strategy/sampler 중심 | loss probe가 있으면 client metric 필요 |
| Privacy/security | update를 숨기거나 noise를 넣음 | 서버/client 양쪽 protocol 필요 | ML 성능보다는 배포 조건 |
| Optimizer foundation | SGD/Adam/SAGA 등 FL의 수학적 기반 | FL method 아님 | local optimizer 선택/해석에 사용 |

주의할 점: FedAvg도 당연히 client에서 local train을 한다. 여기서 "server aggregation only"라고 부르는 뜻은 **client에 별도의 persistent aggregation state가 남지 않고, 서버가 받은 trainable tensor를 평균하는 구조**라는 뜻이다.

## Server-Only와 Local-State 기준 빠른 분리

| 분류 | PPT 알고리즘/주제 | 구현 관점 |
| --- | --- | --- |
| Flower/FedAvg식 server aggregation only | FedSGD, FedAvg, FedAvg with server LR, FedExp | 서버 strategy가 받은 update를 평균/스케일한다. client는 round가 끝나면 별도 FL state가 없다. |
| Server aggregation + client telemetry | FedNova, AFL, q-FFL/q-FedAvg, MaxFL, Power-of-Choice | 서버가 최종 update/selection을 하지만 client loss, local steps, delta norm, benefit metric이 필요하다. |
| Client local objective만 변경 | FedProx | client loss에 proximal term이 들어간다. persistent local parameter는 없어도 된다. |
| Server-only가 아닌 client/server state 방법 | SCAFFOLD, FedVARP | control variate나 variance-reduction memory가 round를 넘어 남는다. 단순 평균 strategy로는 안 된다. |
| Local parameter/model/personalization | FedPer류, global+local fine-tuning, Ditto, clustering, Per-FedAvg, MTL/shared trunk-local head, MMoE | client-local head/model/cluster/adaptation state가 유지된다. ELSA의 diffusion/action/gripper head 분리에 가장 직접적이다. |
| Offline/local delta aggregation 또는 분석 | model merging, task arithmetic, task negation/forgetting | round-wise FL training보다는 env별 adapter/head delta를 합성하거나 분석하는 축이다. |
| Privacy/security protocol | MIA, gradient inversion, DP, secure aggregation, one-time pad, Shamir secret sharing | aggregation 성능 개선이 아니라 update/model leakage를 줄이는 축이다. |

## Lecture별로 나온 알고리즘

| Lecture | 알고리즘/주제 | ELSA에서의 의미 |
| --- | --- | --- |
| lec02 | GD, Batch GD, SGD, Mini-batch SGD, Momentum, Nesterov, AdaGrad, Adam | local optimizer와 batch/LR 해석. FL method라기보다 drift/noise 해석 기반 |
| lec03-04 | SGD convergence, non-convex convergence, SAG, SAGA, SVRG | variance reduction 개념. FedVARP/SCAFFOLD 이해 기반 |
| lec05 | FL framework, FedAvg, client fraction C, local epochs E, batch size B | 현재 Flower baseline의 핵심 축 |
| lec06 | FedSGD, FedAvg, FedProx, SCAFFOLD, FedAvg with server LR, FedExp | heterogeneity/client drift 대응의 1차 후보군 |
| lec07 | computational heterogeneity, generalized FedAvg, FedNova | local step 수가 다를 때 필요한 normalization |
| lec08-09 | partial participation, FedVARP, Power-of-Choice | 일부 env/client만 뽑는 상황의 variance/selection |
| lec10 | fairness, AFL, q-FFL/q-FedAvg, MaxFL, participation incentives | hard env/worst-client 중심 objective |
| lec11 | personalized FL, Ditto, clustering | global-only가 어려운 env/controller 차이에 대응 |
| lec12 | Ditto recap, clustering, meta learning, Per-FedAvg, MTL | quick adaptation과 local personalized model |
| lec13 | MTL, MMoE, MGDA, model merging, task arithmetic | shared trunk/local head, task gradient balancing, delta 분석 |
| lec14-15 | membership inference, gradient inversion, DP, Laplace epsilon-DP, local/global DP, secure aggregation, one-time pad, client dropout handling, Shamir secret sharing | privacy/security protocol. 현재는 낮은 우선순위 |

## 1. Server Aggregation Only 계열

이 그룹은 지금 Flower/FedAvg 구조와 가장 잘 맞는다. 서버 strategy를 바꾸는 것만으로 실험을 시작할 수 있고, client-local persistent state는 필요 없다.

| 알고리즘 | PPT 근거 | 핵심 | ELSA 적용 |
| --- | --- | --- | --- |
| FedSGD | lec06 p.17-p.19, lec07 p.21-p.23 | 매 step/round gradient를 서버가 평균 | 통신량이 커서 robotics trajectory 학습에는 실험 우선순위 낮음 |
| FedAvg | lec05 p.25-p.30, lec06 p.22-p.25 | local epochs 후 model/update 평균 | 반드시 남길 기본 baseline |
| FedAvg + server LR | lec06 p.40, lec07 p.19 | 평균 delta에 서버 learning rate 적용 | shared trunk가 튀는 경우 가장 단순한 안정화 |
| FedExp | lec06 p.42-p.47, lec07 p.21-p.26 | adaptive server learning rate | FedProx 다음으로 붙이기 쉬운 server-side 개선 |

ELSA 해석:

- FedAvg는 "global 하나로 모든 env/controller를 설명할 수 있는가"의 기준선이다.
- FedExp는 client update direction이 round마다 흔들릴 때 서버 step을 자동 조절한다. VolumeDP/DINO shared trunk나 LoRA adapter를 aggregate할 때 먼저 붙일 가치가 있다.
- 이 그룹은 local head personalization을 검증하지 못한다. 따라서 gripper transition/action contract 가설을 증명하기에는 단독으로 부족하다.

현재 상태:

- FedAvg는 구현되어 있다.
- FedProx를 포함한 기존 서버 평균 구조는 유지되어 있다.
- `fede/modules/strategy_blueprints.py`에는 FedExp helper가 있지만 Flower strategy로 연결되지는 않았다.

## 2. Server Aggregation + Client Telemetry 계열

이 그룹은 서버가 최종 결정을 하지만, client가 단순 weight 외 정보를 같이 보내야 한다. 이미 ELSA 코드에는 `pre_train_loss`, `post_train_loss`, `delta_norm_sq`, `local_steps`, `num_batches` telemetry를 넣어두었기 때문에 다음 구현 후보로 현실성이 있다.

| 알고리즘 | PPT 근거 | 필요한 client 반환값 | ELSA 적용 |
| --- | --- | --- | --- |
| FedNova | lec07 p.49-p.52, lec08 p.24-p.27 | local steps, effective tau/normalizer | env별 trajectory 수/epoch/속도가 다를 때 objective inconsistency 보정 |
| AFL | lec10 p.18-p.23, lec11 p.3-p.4 | client/domain loss | worst-env 성능을 직접 올리는 minimax 계열 |
| q-FFL/q-FedAvg | lec10 p.25-p.28, q-FFL paper p.5-p.6 | client loss, delta norm, optional Hessian-like term | 평균 성능과 hard-env 성능 사이를 q로 조절 |
| MaxFL | lec10 p.41-p.42, lec11 p.19-p.20 | client benefit/gain 또는 loss threshold | "이 env가 FL에 참여해서 실제 이득을 보는가"를 objective로 표현 |
| Power-of-Choice | lec08 p.44, lec09 p.24, lec10 p.6 | candidate client loss/probe | 고손실 env를 더 자주 뽑아 gripper failure를 빠르게 줄임 |

ELSA 해석:

- Ralph/ELSA는 env별 실패 양상이 평균 loss에 묻히기 쉽다. 특히 gripper open/close transition은 전체 trajectory loss 중 비중이 작다.
- q-FedAvg/AFL/MaxFL을 쓰려면 loss를 그냥 전체 MSE/BCE 평균으로 두면 안 된다. gripper transition-weighted BCE, action event loss, post-transition window loss 같은 per-client metric을 따로 반환해야 한다.
- FedNova는 공정성보다 computational heterogeneity 대응이다. local batch 수가 env마다 다르거나 중간에 실패/skip이 생기면 FedAvg보다 먼저 검토할 만하다.

권장 구현 순서:

1. FedExp: 서버 strategy만 추가.
2. FedNova: 이미 `local_steps`가 있으므로 aggregation normalizer를 추가.
3. q-FedAvg: `metrics-probe-batches`로 pre-loss를 안정화한 뒤 적용.
4. AFL/MaxFL: hard-env 목적이 명확해진 뒤 적용.
5. Power-of-Choice: client sampling API까지 건드려야 하므로 그 다음.

## 3. Client Local Objective / Correction 계열

이 그룹은 서버 평균만 바꿔서는 구현할 수 없다. client train loop가 달라진다.

| 알고리즘 | PPT 근거 | client 쪽 변화 | ELSA 적용 |
| --- | --- | --- | --- |
| FedProx | lec06 p.29-p.31, lec07 p.13 | local loss에 global anchor proximal term 추가 | 현재 가장 중요한 drift baseline |
| SCAFFOLD | lec06 p.35-p.38, lec07 p.15 | server/client control variate로 local gradient 보정 | camera/object/task heterogeneity가 심한 경우 FedProx보다 강한 correction |
| Soft parameter sharing / anchor regularization | lec13 MTL section | client model을 anchor에 regularize | Ditto/FedProx/FedPer의 일반화로 해석 가능 |

ELSA 해석:

- FedProx는 이미 넣어둔 방향이 맞다. 다만 FedProx를 전체 parameter에 걸면 local head personalization과 충돌할 수 있으므로 shared aggregated surface에만 걸어야 한다.
- SCAFFOLD는 control variate tensor가 aggregated parameter surface와 같은 shape를 가져야 한다. FedPer처럼 local head를 제외하면 control variate도 shared body에만 둬야 한다.
- gripper event head를 local로 둘 경우, FedProx는 shared representation에는 drift 억제를 걸고 local head에는 자유도를 남기는 구조가 더 타당하다.

현재 상태:

- `fedprox`는 구현되어 있다.
- `fedprox_fedper_head`에서 FedProx anchor가 aggregated/shared parameter에만 적용되도록 바꿔두었다.
- SCAFFOLD는 아직 없다. 다음 구현 시 client별 `c_i`와 server `c`를 local state path에 맞춰 저장해야 한다.

## 4. Client-Local Persistent State / Local Aggregation 계열

사용자가 말한 "server aggregation만 아닌 방법"의 핵심은 이 그룹이다. 서버가 global model을 갖더라도, client에는 round를 넘어 유지되는 local parameter/model/state가 남는다.

| 알고리즘/주제 | PPT 근거 | 남는 상태 | ELSA 적용 |
| --- | --- | --- | --- |
| Local-only baseline | lec11 personalized FL section | client별 모델 전체 | FL이 정말 이득인지 비교하는 필수 baseline |
| Global + local fine-tuning | lec11 personalized FL section | global init 후 local adapted model | 같은 env에서 personalized eval 필요 |
| Ditto | lec11 p.27-p.29, lec12 p.7-p.9 | global model + personalized local model | strong personalized FL baseline |
| Clustering / model-based clustering | lec11 p.33-p.36, lec12 p.13-p.16 | cluster별 model/head | env를 visual/controller regime으로 묶는 실험 |
| Meta learning / Per-FedAvg | lec12 p.18-p.21, lec13 p.7-p.10 | 빠른 adaptation을 위한 init과 inner-loop | 새 object/camera에 빠르게 적응하는 정책 |
| MTL/shared trunk/local heads | lec12 p.23, lec13 p.12-p.25 | shared body + task/client head | VolumeDP/DINO shared trunk와 action/diffusion/gripper head 분리에 가장 직접적 |
| MMoE | lec12 p.36-p.39, lec13 p.25-p.35 | shared experts + task/client gates | env/task conditioned controller; 구현량 큼 |
| MGDA/MOO | lec12 p.39, lec13 p.35 | local 또는 server gradient combiner | action loss, diffusion loss, gripper event loss 균형 |
| Model merging / task arithmetic | lec13 p.36-p.41, lec14 p.14 | offline/local delta 조합 | env별 LoRA/head delta 분석과 one-shot fusion |

ELSA 해석:

- 현재 우리가 만든 `fedper_head` / `fedprox_fedper_head`는 이 축의 첫 구현이다.
- 추천 parameter split은 다음과 같다.

| Surface | 권장 상태 | 이유 |
| --- | --- | --- |
| DINO/VolumeDP/FLAME shared visual-spatial trunk | server aggregated | env 간 공유해야 하는 representation |
| LoRA shared adapter 일부 | server aggregated 또는 cluster-local | visual regime이 비슷하면 공유, 다르면 cluster |
| diffusion/action decoder head | client-local 후보 | controller/action contract가 env별로 다를 수 있음 |
| gripper event head | client-local 또는 transition-weighted global ablation | transition timing이 env별로 다르면 local이 유리 |
| normalization/statistics류 | client-local 후보 | camera/object distribution shift에 민감 |

중요한 평가 조건:

- client-local head를 쓰는 방법은 global checkpoint 하나만 평가하면 안 된다.
- 같은 `partition_id/env_id/run_tag`의 local state를 복원한 personalized eval path가 필요하다.
- 따라서 `fedper_head` 계열은 "제출용 global single model"과 "분석용 personalized model"을 분리해서 보고해야 한다.

## 5. Client Selection / Participation

| 알고리즘/주제 | PPT 근거 | 서버-only인가 | ELSA 적용 |
| --- | --- | --- | --- |
| Client fraction C | lec05 p.28 | sampling은 서버 쪽 | 모든 env를 매 round 못 돌릴 때 기본 변수 |
| Partial participation convergence | lec08 p.30-p.33, lec09 p.10-p.13 | 분석/서버 sampling | launched/active worker 제한과 직접 관련 |
| FedVARP | lec08 p.36-p.39, lec09 p.16-p.19 | 서버 memory 필요 | partial participation variance가 크면 적용 |
| Power-of-Choice / adaptive d | lec08 p.44-p.46, lec09 p.24-p.26, lec10 p.6 | 서버 sampler + loss probe | hard-env를 더 자주 뽑아 실패 env 개선 |
| Participation incentives / MaxFL | lec10 p.38-p.42 | 서버 objective + client benefit metric | client가 FL 참여로 손해보는지 분석 |

ELSA 해석:

- 지금 active worker가 제한되는 harness 상황에서는 partial participation을 단순 시스템 문제로만 보면 안 된다. 학습 objective 자체가 바뀐다.
- Power-of-Choice는 hard-env를 빠르게 고치는 데 매력적이지만, high-loss client만 자주 뽑으면 평균/일반화가 흔들릴 수 있다.
- 먼저 모든 client의 loss/transition failure metric을 안정적으로 수집한 뒤 selection을 바꾸는 것이 맞다.

## 6. Privacy / Security

이 그룹은 지금 연구의 주 병목을 직접 풀지는 않는다. 다만 federated learning 논문/수업 맥락에서는 빠질 수 없는 축이다.

| 알고리즘/주제 | PPT 근거 | 어디서 적용 | ELSA 적용 판단 |
| --- | --- | --- | --- |
| Membership inference attack | lec15 p.4-p.5 | black-box output 또는 white-box gradient/parameter 관찰 | global model을 round마다 공유할 때 생기는 leakage audit |
| Gradient inversion attack | lec14 p.23-p.24, lec15 p.6-p.7 | 공격/진단 | 로봇 trajectory/update가 민감할 때 위험 분석 |
| Differential Privacy | lec14 p.26-p.37, lec15 p.9-p.20 | client 또는 server noise | 현재 benchmark 성능 실험에는 낮은 우선순위 |
| Laplace epsilon-DP mechanism | lec15 p.14-p.15 | DP noise mechanism | 문서/보안 설명용. 지금 성능 실험 우선순위는 낮음 |
| Local DP | lec14 p.33,p.36-p.37, lec15 p.16,p.19-p.20 | client가 보내기 전 noise | utility 손실 큼 |
| Global DP | lec14 p.34-p.35, lec15 p.17-p.18 | server aggregation 후 noise | trusted server이면 상대적으로 현실적 |
| Secure Aggregation | lec14 p.38, lec15 p.22-p.35 | encrypted/masked sum | 개별 client update를 서버도 못 보게 함 |
| One-time Pad | lec15 p.23-p.26 | pairwise mask | secure aggregation building block |
| Client dropout handling | lec15 p.27 | secure aggregation protocol | client가 중간 탈락할 때 pairwise mask를 복구해야 하는 문제 |
| Shamir Secret Sharing | lec15 p.28-p.34 | dropout-robust secret recovery | secure aggregation building block |

ELSA 적용 판단:

- 단일 실험 서버/단일 연구실 benchmark에서는 DP/SecAgg가 먼저가 아니다.
- 다기관/실제 로봇 로그 공유 시나리오로 논문을 확장할 때는 "local head/personalized state는 외부로 안 나가고, shared trunk update만 secure aggregation" 같은 구조가 설득력 있다.
- DP를 넣는다면 gripper transition 같은 희소 event signal이 noise에 약하므로 event metric 실험을 먼저 끝낸 뒤 넣어야 한다.
- secure aggregation은 개별 update를 숨기지만 서버가 개별 client metric을 볼 수 없게 만들 수 있다. q-FedAvg/AFL/Power-of-Choice처럼 client별 loss가 필요한 방법과 동시에 쓰려면 metric 공개 범위를 따로 설계해야 한다.

## 7. Optimizer / Variance Reduction Foundations

이들은 PPT에 나온 알고리즘이지만 FL aggregation 방식 자체는 아니다.

| 알고리즘 | PPT 근거 | ELSA에서 봐야 할 점 |
| --- | --- | --- |
| GD / Batch GD | lec02 p.8,p.17 | 이론적 기준선 |
| SGD / Mini-batch SGD | lec02 p.19-p.26 | local batch size와 local step 수가 FL update bias/noise를 결정 |
| Momentum / Nesterov | lec02 p.31-p.35 | local drift를 키울 수도 있으므로 hetero 환경에서 조심 |
| AdaGrad / Adam | lec02 p.38-p.45 | local optimizer로 안정적이나, FL method 이름과 분리해 ablation |
| SAG / SAGA / SVRG | lec04 p.30-p.34 | FedVARP/SCAFFOLD의 variance-reduction 해석 기반 |

ELSA 적용 판단:

- local optimizer ablation은 필요하지만 논문의 주 novelty로 세우기는 약하다.
- 현재는 optimizer보다 "어떤 parameter를 공유하고 어떤 state를 local에 둘 것인가"가 더 중요하다.

## 현재 코드와의 매핑

| 코드 | 현재 의미 |
| --- | --- |
| `federated_elsa_robotics/fl_method_registry.py` | method를 `server_strategy`, `client_update`, `local_regularizer`, `parameter_scope`, `client_state` 축으로 분해 |
| `federated_elsa_robotics/parameter_surfaces.py` | aggregated parameter와 local-only parameter 분리, local-only state 저장/복원 |
| `federated_elsa_robotics/task.py` | config-aware get/set weights, FedProx anchor를 aggregated surface에만 적용 |
| `federated_elsa_robotics/client_app.py` | loss/delta/local_steps telemetry와 local-only parameter state load/save |
| `federated_elsa_robotics/server_app.py` | telemetry aggregation, manifest 저장, config-aware server weights, strategy hyperparameter 연결 |
| `federated_elsa_robotics/server_aggregation.py` | FedAvg/FedExp/FedNova/q-FedAvg/AFL/MaxFL server aggregation |
| `experiments/fl_dinov3_diffusion_lora4_jvdirect_fedprox_fedper_head.yaml` | FedProx + local diffusion/action/gripper head template |
| `scripts/eval_flower_personalized_local_state.py` | server checkpoint + local head state personalized offline eval |
| `fede/modules/strategy_blueprints.py` | FedExp, FedNova, AFL, q-FFL, MaxFL helper 수식/weight prototype/reference |

현재 구현으로 가능한 분류:

- **server aggregation only**: `fedavg`, `fedprox`의 서버 aggregation 자체.
- **client objective correction**: `fedprox`.
- **client-local persistent parameter**: `fedper_head`, `fedprox_fedper_head`.
- **telemetry-based server strategy implemented**: `fedexp`, `fednova`, `qfedavg`, `afl`, `maxfl`.
- **not implemented**: SCAFFOLD, FedVARP, Ditto full personalized model, Power-of-Choice sampler, DP/SecAgg.

## ELSA 우선순위

### P0: 지금 바로 실험/검증할 축

1. `fedavg`: global-only baseline.
2. `fedprox`: client drift baseline.
3. `fedprox_fedper_head`: shared body + local diffusion/action/gripper head.
4. transition-weighted gripper BCE metric을 client telemetry로 분리.
5. personalized eval path: local state 복원 후 same-env 평가.

### P1: telemetry가 안정되면 바로 추가할 축

1. FedExp: adaptive server LR.
2. FedNova: local_steps/num_batches가 client별로 다를 때.
3. q-FedAvg: hard-env/event-loss를 더 세게 반영.
4. AFL/MaxFL: worst-env 또는 participation benefit 분석.

### P2: 논문 스토리를 키울 때 추가할 축

1. SCAFFOLD: control variate로 client drift 보정.
2. FedVARP: partial participation variance reduction.
3. Ditto: full personalized local model.
4. Clustering/HyperCluster: env regime별 model/head.
5. MMoE/MGDA: architecture/objective-level multi-tasking.
6. Task arithmetic/model merging: env-specific delta 분석.

### P3: 배포/보안 스토리

1. Membership inference and gradient inversion risk audit.
2. Global DP after aggregation.
3. Secure aggregation with local personalized heads hidden.

## 우리 아이디어로 정리한 실험 가설

### 가설 A: Server-only aggregation은 gripper event를 평균에 묻는다

- 비교: `fedavg` vs `fedprox`.
- metric: action loss 전체 평균 말고 transition window BCE, open/close timing error, post-transition control error.
- 예상: global-only는 평균 trajectory는 좋아져도 rare event가 늦게 개선된다.

### 가설 B: Shared representation은 global, controller contract는 local이 낫다

- 비교: `fedprox` vs `fedper_head` vs `fedprox_fedper_head`.
- shared: DINO/VolumeDP/FLAME-like visual-spatial trunk.
- local: diffusion/action/gripper heads.
- 예상: local head가 env/controller-specific timing을 보존한다.

### 가설 C: hard-env weighted aggregation은 평균 성능보다 실패 env 회복을 빠르게 한다

- 비교: FedAvg/FedProx vs q-FedAvg/AFL/MaxFL.
- 필요한 준비: client loss telemetry가 event-aware여야 한다.
- 예상: q/AFL/MaxFL은 worst-env success를 올리지만 평균/안정성 trade-off가 있다.

### 가설 D: local step heterogeneity를 보정하지 않으면 aggregation bias가 생긴다

- 비교: FedAvg/FedProx vs FedNova.
- 조건: env별 batch 수, trajectory 길이, fail/skip 비율이 다를 때.
- 예상: FedNova가 round-to-round stability를 개선한다.

### 가설 E: partial participation 자체가 objective를 바꾼다

- 비교: uniform sampling vs Power-of-Choice vs FedVARP.
- 조건: worker limit 때문에 일부 env만 학습하는 상황.
- 예상: Power-of-Choice는 hard-env를 빠르게 개선하지만 selection bias가 있고, FedVARP는 variance 안정화 쪽이다.

## 다음 구현 제안

1. `strategies.py`에 FedExpStrategy를 추가한다.
   - 입력: aggregated delta, optional `delta_norm_sq`.
   - 목표: server LR 자동 조절.

2. FedNovaStrategy를 추가한다.
   - 입력: client `local_steps`, `num_examples`, delta.
   - 목표: heterogeneous local epoch/step 보정.

3. q-FedAvg/AFL/MaxFL strategy family를 추가한다.
   - 입력: `pre_train_loss` 또는 event-aware client loss.
   - 목표: hard-env/fairness objective.

4. personalized eval path를 추가한다.
   - 입력: `checkpoint-root`, `run-tag`, `partition_id`, `env_id`.
   - 목표: local-only head를 복원해서 평가.

5. SCAFFOLD state schema를 설계한다.
   - server: global control variate.
   - client: per-client control variate.
   - parameter surface: aggregated tensors only.

## 최종 판단

현재 우리가 이미 구현한 `fedprox_fedper_head`는 PPT taxonomy상 **server aggregation only를 넘어서는 첫 번째 의미 있는 구조**다. 단순히 FedAvg/FedProx server strategy를 바꾸는 수준이 아니라, shared representation과 local controller/action head의 계약을 분리한다.

따라서 다음 문서/실험의 중심 문장은 이렇게 잡는 것이 좋다.

> ELSA/Ralph류 로봇 imitation FL의 병목은 vision backbone 평균화가 아니라, temporal gripper event와 action decoder contract가 client/env별로 어긋나는 데 있다. 따라서 server-only aggregation baseline 위에 client-local head/personalized state와 event-aware fairness aggregation을 결합해야 한다.
