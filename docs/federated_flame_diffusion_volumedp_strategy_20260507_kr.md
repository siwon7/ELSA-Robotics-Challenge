# Federated FLAME / Diffusion / VolumeDP 전략 정리

작성일: 2026-05-07 KST

이 문서는 FLAME류 manipulation FL 환경에서 `DINO/LoRA + diffusion decoder`,
`Depth`, `VolumeDP-like volumetric policy`를 쓸 때 무엇을 federated로 공유하고,
무엇을 local/frozen으로 남겨야 하는지 정리한 실행 메모다. 현재 큐
`ralph_fill4_power_moved_20260507`는 건드리지 않았다.

## 1. 한 줄 결론

지금 연구의 좋은 방향은 "더 강한 FL aggregator"보다 먼저 "공유할 파라미터
표면을 작고 의미 있게 만드는 것"이다. 즉 frozen foundation backbone은 고정하고,
LoRA/projector/volumetric token module/diffusion decoder만 trainable-only로
모은 뒤, FedAvg와 작은 FedProx grid를 반드시 같이 비교한다.

## 2. 기준 논문에서 얻는 제약

- FLAME은 조작 정책 데이터를 여러 환경/client에 분산해 학습하는 benchmark이며,
  raw data를 중앙에 모으지 않는 federated policy learning의 어려움을 직접
  다룬다. 논문상 핵심 난점은 data가 non-IID이고 환경 visual shift가 크다는
  점이다. 우리 문제에서는 color/background/camera/view shift와 task별 action
  timing mismatch가 여기에 해당한다.
- FedAvg는 local update를 평균하는 기본선이다. communication round를 줄이는
  실용 baseline이지만, non-IID 환경에서는 client drift가 생길 수 있다.
- FedProx는 FedAvg local objective에 global model과의 거리 penalty를 추가해
  heterogeneity를 완화한다. 다만 proximal strength가 너무 크면 local adaptation을
  막는다. 우리 기존 pilot에서 `DINO LoRA4 + diffusion + JV direct`는 FedAvg가
  FedProx보다 좋았으므로 FedProx를 default winner로 두면 안 된다.
- SCAFFOLD는 client drift를 control variate로 보정한다. 구현은 무겁지만,
  FedAvg/FedProx 모두 흔들릴 때 다음으로 볼 가치가 가장 높다.
- FedNova는 client별 local update 횟수나 compute가 크게 다를 때 objective
  inconsistency를 줄이는 쪽이다. 현재처럼 `local_epochs`와 client budget을
  맞춘 실험에서는 우선순위가 낮다.
- FedBN은 feature-shift non-IID에서 BatchNorm 통계를 client-local로 남긴다.
  현재 주력 DINO/Transformer/LayerNorm 계열에는 직접 이득이 작고, CNN baseline
  비교나 BN module이 실제로 있을 때만 의미가 있다.
- Diffusion Policy는 action distribution을 denoising diffusion process로
  표현하므로 multimodal action과 high-dimensional action에 유리하다. 대신
  federated 상황에서는 decoder update가 client별 action style/timing에 민감해진다.
- VolumeDP는 2D visual feature를 3D volumetric representation으로 lift하고,
  task-relevant voxel을 spatial token으로 줄인 뒤 multi-token decoder로 action을
  예측하는 구조다. FL에서는 이 3D lift가 camera/extrinsic convention에 매우
  민감하므로 coordinate contract가 깨지면 오히려 drift를 키운다.

## 3. 현재 레포 구현 상태

현재 `federated_elsa_robotics` 구현은 다음 상태다.

- 서버 aggregation은 strategy preset으로 선택한다.
  - `TrainableOnlyFederatedStrategy`가 `ServerSideAggregator`를 통해
    `fedavg`, `fedexp`, `fednova`, `qfedavg`, `afl`, `maxfl`을 처리한다.
  - checkpoint 저장 시 전체 state dict를 저장하지만, 통신/aggregation 대상은
    `get_weights(agent, config)`가 반환하는 aggregated parameter surface다.
  - `fedper_head`, `fedprox_fedper_head`는 diffusion/gripper/action head를
    client-local state로 저장하고, 서버에는 shared surface만 보낸다.
- FedProx는 server strategy가 아니라 client-side local regularizer다.
  - `federated_elsa_robotics/task.py:111-119`에서
    `loss + 0.5 * prox_mu * ||w_local - w_global||^2`를 더한다.
  - `global_trainable_params`는 client round 시작 시점의 trainable params clone이다.
- trainable-only aggregation은 `param.requires_grad`와 method preset의
  local-only keyword를 함께 본다.
  - `federated_elsa_robotics/parameter_surfaces.py`가 aggregated tensor와
    client-local tensor를 나눈다.
  - 따라서 frozen DINO/Depth가 진짜 `requires_grad=False`면 통신되지 않고,
    FedPer 계열에서는 지정된 head도 서버 aggregation에서 빠진다.
- client model shape은 sample batch에서 잡는다.
  - `federated_elsa_robotics/client_app.py:148-165`가 sample image channel,
    low_dim, action_dim을 읽어 `Agent`를 만든다.
- server model shape은 일부 보강했다.
  - 2026-05-07 패치로 server도 `image_channels`를
    `get_expected_image_channels(conf)`에서, `low_dim_state_dim`을
    `infer_low_dim_state_dim(conf)`에서 잡는다.
  - 아직 `image_size=(128, 128)`는 기존 dataset transform contract에 기대고 있다.
    다른 resize/crop을 넣는 FL config가 생기면 server/client shape manifest로
    먼저 비교해야 한다.
- 레포 registry상 backbone aggregation 의도는 명확하다.
  - `elsa_learning_agent/model_registry.py:25-31`: frozen DINO는 projector +
    policy head만 trainable.
  - `model_registry.py:49-55`: VolumeDP-lite는 DINO LoRA/projectors +
    volumetric token module + policy head.
  - `model_registry.py:65-71`: VolumeDP-full은 DINO LoRA + depth/voxel projectors
    + softmax token learner + policy head.

## 4. 무엇을 aggregate할 것인가

### 4.1 기본 원칙

- raw RGB, raw depth, camera intrinsics/extrinsics, per-client voxel volume은 절대
  공유하지 않는다.
- frozen backbone weight는 공유할 필요가 없다. byte-identical checkpoint를
  deployment artifact로 맞추고, FL 통신에서는 제외한다.
- aggregate 대상은 "client마다 학습되지만 global policy에 필요한 작은 모듈"이다.
  현재 후보는 LoRA, visual/projector MLP, proprio/state encoder, volumetric token
  learner, diffusion decoder, gripper/event head다.
- personalization은 global evaluation이 single global checkpoint라면 후순위다.
  client identity를 test time에 쓸 수 없으면 local head는 paper contribution과
  evaluation contract가 어긋날 수 있다.

### 4.2 구조별 권장 aggregation surface

| 구조 | 공유/aggregate | local/frozen | 권장 FL |
|---|---|---|---|
| CNN + MLP baseline | 전체 encoder/head | raw data | FedAvg baseline |
| Frozen DINO + MLP | projector, state encoder, action head | DINO base | FedAvg, FedProx small grid |
| Frozen DINO + diffusion | projector, diffusion decoder, state encoder, gripper head | DINO base | FedAvg 필수, FedProx 보조 |
| DINO LoRA + diffusion | LoRA, projector, diffusion decoder, state encoder, gripper/event head | DINO base | FedAvg vs FedProx, local_epochs 짧게 |
| DINO + Depth + diffusion | DINO/Depth projectors, fusion, decoder | DINO/Depth base, generated depth maps | FedAvg 먼저, depth compute deterministic 유지 |
| VolumeDP-lite | LoRA/projector, voxel/spatial token module, diffusion decoder | DINO base, local camera metadata | FedAvg/FedProx, grid 작게 |
| VolumeDP-full | DINO LoRA, depth/voxel projectors, soft token learner, multi-token decoder | DINO/Depth base, `K/T`, depth maps, voxel volume | same-env sanity 후 FL |

## 5. FLAME 환경에서의 판단

FLAME류 환경은 client가 단순 class imbalance를 갖는 게 아니라, 같은 task라도
visual/camera/background/action timing이 달라지는 non-IID 문제다. 따라서 다음
순서가 안전하다.

1. 같은 환경에서 action/controller contract를 먼저 안정화한다.
   현재 Ralph 계열 결과처럼 실패 원인이 gripper transition/event grounding이면
   federated aggregation을 바꿔도 온라인 성공률이 바로 오르지 않는다.
2. FL은 success가 0이 아닌 same-env policy family에만 붙인다.
   zero-success policy를 federated로 평균하면 drift 분석이 불가능하다.
3. FedAvg를 항상 남긴다.
   기존 `DINO LoRA4 + diffusion + JV direct` pilot은 FedAvg가 FedProx보다 좋았다
   (`slide`: 0.16 vs 0.04, `close_box`: 0.34 vs 0.22).
4. FedProx는 `prox_mu=1e-3` 하나로 결론내지 말고 grid로 본다.
   추천 grid: `0`, `1e-4`, `3e-4`, `1e-3`, `3e-3`.
5. SCAFFOLD는 "FedAvg도 흔들리고 FedProx도 손해"인 경우에 구현한다.
   지금 당장은 aggregation surface와 action contract를 줄이는 것이 먼저다.

## 6. Diffusion decoder가 있을 때

### 6.1 왜 조심해야 하나

Diffusion decoder는 단순 regression head보다 action mode와 timing을 더 잘 담지만,
federated averaging에서는 client별 action style이 decoder weight에 직접 섞인다.
특히 gripper open/close transition, receding horizon execute step, joint velocity
servo gain이 client마다 다르면 decoder 평균이 중간 행동을 만들 수 있다.

### 6.2 권장 방식

- 첫 FL diffusion 실험은 `local_epochs=1, 3, 5`를 우선 본다.
  diffusion decoder는 local overfit이 빠를 수 있으므로 10 epoch 이상은 뒤로 미룬다.
- FedAvg와 FedProx를 같이 둔다. FedProx가 "항상 좋다"가 아니라 drift가 강할 때만
  도움이 되는 regularizer로 취급한다.
- diffusion EMA를 추가한다면 client EMA를 그냥 평균하지 않는다.
  더 안전한 방식은 server가 online aggregated weight를 저장하고, 별도 server-side
  EMA를 유지하거나 EMA 없이 평가하는 것이다.
- loss scale을 고정한다.
  diffusion MSE, gripper BCE, EE aux loss를 섞으면 FedProx penalty의 상대 크기가
  바뀐다. `prox_mu` 비교 시 loss 구성은 고정해야 한다.
- action contract를 log에 남긴다.
  `action_type`, `action_chunk_len`, `receding_horizon_execute_steps`,
  `gripper_eval_mode`, threshold/hysteresis, servo gain은 checkpoint metadata에
  함께 있어야 FL 결과를 해석할 수 있다.
- gripper/event head는 처음에는 global aggregate한다.
  다만 transition-weighted BCE나 event head가 들어간 뒤 client별 gripper timing이
  크게 다르면, "shared visual/diffusion + local small gripper calibration"은
  별도 ablation으로 볼 수 있다.

## 7. LoRA가 있을 때

LoRA는 frozen foundation model 위에 작은 trainable low-rank matrix를 얹는 방식이라,
FL에서 통신량과 client drift를 줄이기 좋다. 다만 LoRA rank와 target module이 바뀌면
parameter shape이 달라지므로 한 FL run 안에서는 반드시 고정해야 한다.
FLoRA 계열 논문은 heterogeneous LoRA adapter를 단순 평균하면 수학적으로 맞지 않는
noise가 생길 수 있음을 지적한다. 우리 실험에서는 rank/target을 고정한 homogeneous
LoRA부터 시작하고, heterogeneous rank는 후순위로 둔다.

권장:

- `rank=4`와 `rank=8`을 먼저 비교한다.
- `qkv,proj` target은 유지하되, trainable parameter audit을 매 run 저장한다.
- DINO base weight는 frozen/byte-identical이어야 한다.
- FedProx는 LoRA surface가 작을수록 과하게 constraining할 수 있으므로, 기존 pilot
  결과처럼 FedAvg가 이길 가능성을 열어둔다.

## 8. VolumeDP-like 구조일 때

### 8.1 핵심 위험

VolumeDP의 장점은 2D feature를 3D coordinate로 맞추는 것이다. FL에서는 이 장점이
동시에 위험이 된다. client마다 camera calibration, extrinsic convention, crop/resize,
depth scale, workspace bounds가 조금씩 다르면 같은 voxel index가 서로 다른 물리
공간을 의미할 수 있다.

### 8.2 권장 contract

- 모든 client가 같은 world-frame definition을 써야 한다.
- `K`, `T`, crop/resize transform, depth scale, workspace bounds,
  `volumedp_grid_shape`, token count를 checkpoint config에 저장한다.
- `K/T`와 per-client voxel tensor는 local-only다.
- aggregate는 voxel lift/projector, token learner, diffusion decoder만 한다.
- camera-free VolumeDP-lite는 빠른 ablation으로 좋지만, 논문상 VolumeDP의 주장을
  제대로 검증하는 최종 형태는 아니다. "geometry가 빠진 token learner" baseline으로
  취급한다.
- grid는 처음에 `8^3` 또는 `16^3`로 제한한다. `32^3` 이상은 FL 통신/메모리/속도
  모두에서 early experiment에 부적합하다.
- VolumeDP-full은 same-env에서 non-zero success가 확인된 뒤 FL로 올린다.
  same-env에서 약하면 FL에서 원인 분리가 불가능하다.

### 8.3 VolumeDP aggregation variants

- V1 Global lightweight:
  aggregate `{LoRA, fusion/projector, voxel lift, spatial-token learner,
  diffusion decoder}`. 가장 단순하고 현재 trainable-only aggregation과 잘 맞는다.
- V2 Shared geometry / local action:
  aggregate `{LoRA, volumetric/token module}`, decoder는 local. global checkpoint
  평가에는 불리하므로 personalization 평가가 가능할 때만 본다.
- V3 Shared DINO/LoRA / local camera-aware geometry:
  aggregate `{LoRA, decoder}`, camera-aware voxel module은 local. extrinsic noise가
  큰 상황에서 비교 가치가 있다.
- V4 Staged:
  먼저 V1로 global representation을 만들고, 마지막에 decoder/head를 local
  fine-tune한다. paper용으로는 좋지만 현재 harness 우선순위는 아니다.

## 9. 지금 당장 미리 할 일

학습 큐를 건드리지 않고 할 수 있는 선행 작업은 다음 순서다.

1. FL server/client shape manifest 추가
   - `image_channels`와 `low_dim_state_dim`은 2026-05-07에 config 기반으로 보강했다.
   - 같은 패치로 server는 trainable tensor 이름, shape, param count를
     `*.trainable_manifest.json`으로 저장하고 client는 동일 summary를 로그에 찍는다.
   - 남은 일은 server/client manifest를 자동 비교해서 mismatch를 round 0 전에
     hard fail시키는 것이다.
2. trainable parameter audit script 추가
   - 서버 manifest 저장은 들어갔다. 별도 CLI script를 만들면 FL run 없이 config만
     audit할 수 있어 proposal 표 작성이 쉬워진다.
   - VolumeDP/LoRA 실험에서 "정말 작은 surface만 aggregate했는지" 증명하는 데 필요하다.
3. FL config matrix 생성
   - `fedavg`: `prox_mu=0`
   - `fedprox`: `prox_mu in {1e-4, 3e-4, 1e-3, 3e-3}`
   - `local_epochs in {1,3,5}`
   - first surface: `DINO LoRA4/8 + diffusion + JV direct`
4. VolumeDP FL preflight만 먼저 만든다.
   - config validation, one-client one-batch forward/loss, trainable param audit,
     no rollout.
5. TGAC 결과가 끝난 뒤에만 FL online eval을 붙인다.
   - gripper transition 문제를 해결하지 않고 FL로 넘어가면 실패 원인이 aggregator인지
     action contract인지 분리되지 않는다.

## 10. 추천 실험 순서

### Stage A: current queue 결과 수집 후

- Ralph fill4 queue 결과에서 `success`, gripper transition metrics, hysteresis
  effect를 먼저 정리한다.
- `gripper_transition_weight`, hysteresis, event head 순서의 TGAC ablation을 확정한다.

### Stage B: FL diffusion baseline

- target: non-zero same-env family
- architecture: `DINO LoRA4/8 + diffusion + JV direct`
- methods:
  - FedAvg, `local_epochs=1/3/5`
  - FedProx, `prox_mu=1e-4/3e-4/1e-3/3e-3`, `local_epochs=3 or 5`
- eval:
  - env400..409 같은 held-out visual env
  - stochastic diffusion seed fixed set
  - success mean/std, gripper transition F1, action magnitude stats

### Stage C: DINO+Depth

- same-env에서 task별 winner가 확실할 때만 FL로 이동한다.
- Depth는 frozen/local deterministic feature generator로 둔다.
- aggregate는 DINO/Depth projectors, fusion, diffusion decoder.

### Stage D: VolumeDP-lite/full

- 먼저 same-env VolumeDP-lite/full sanity.
- 그 다음 one-task FL smoke.
- 마지막에 FLAME-style multi-env client sampling으로 이동한다.

## 11. 현재 가설

가장 방어 가능한 paper hypothesis는 다음이다.

> Federated manipulation에서 성능을 가르는 것은 aggregator 자체보다 aggregation
> surface다. Frozen visual foundation model + small LoRA + geometry-aware
> volumetric token module + diffusion decoder로 공유 표면을 제한하면, 강한
> non-IID 환경에서도 FedAvg가 SCAFFOLD류 heavy correction과 경쟁할 수 있다.

이 가설은 좋은 점이 있다. FedAvg가 이겨도 논문 스토리가 살고, SCAFFOLD가 이겨도
"어떤 heterogeneity가 아직 남았는가"를 분석할 수 있다. 반대로 바로 SCAFFOLD부터
구현하면 실패했을 때 action contract, decoder drift, volumetric coordinate mismatch,
aggregator 중 무엇이 원인인지 분리하기 어렵다.

## 12. 참고한 primary sources

- FLAME: Federated Learning Across Manipulation Environments, arXiv:2503.01729
  https://arxiv.org/abs/2503.01729
- Federated Imitation Learning: A Novel Framework for Cloud Robotic Systems
  with Heterogeneous Sensor Data, ICRA 2020
  https://ram-lab.com/papers/2020/ICRA20_3471_FI.pdf
- FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data,
  arXiv:1602.05629
  https://arxiv.org/abs/1602.05629
- FedProx: Federated Optimization in Heterogeneous Networks, arXiv:1812.06127
  https://arxiv.org/abs/1812.06127
- SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,
  arXiv:1910.06378
  https://arxiv.org/abs/1910.06378
- FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated
  Optimization, arXiv:2007.07481
  https://arxiv.org/abs/2007.07481
- FedBN: Federated Learning on Non-IID Features via Local Batch Normalization,
  arXiv:2102.07623
  https://arxiv.org/abs/2102.07623
- LoRA: Low-Rank Adaptation of Large Language Models, arXiv:2106.09685
  https://arxiv.org/abs/2106.09685
- FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous
  Low-Rank Adaptations, arXiv:2409.05976
  https://arxiv.org/abs/2409.05976
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,
  arXiv:2303.04137
  https://arxiv.org/abs/2303.04137
- 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple
  3D Representations, arXiv:2403.03954
  https://arxiv.org/abs/2403.03954
- Training Diffusion Models with Federated Learning, arXiv:2406.12575
  https://arxiv.org/abs/2406.12575
- VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning,
  arXiv:2603.17720
  https://arxiv.org/abs/2603.17720
