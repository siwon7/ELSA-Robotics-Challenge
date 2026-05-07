# FL Method 축 분해와 Non-Server-Only 구현 메모

작성일: 2026-05-07

## 왜 바꿨나

기존 ELSA FL은 대부분 서버 aggregation 축만 열려 있었다. 서버가 client weight를 받고 FedAvg로 평균하는 구조라서 FedAvg, FedExp, q-FedAvg, FedNova 같은 서버/metric 중심 방법은 붙이기 쉽지만, 다음 계열은 구조적으로 다르다.

- FedProx: client local objective를 바꾼다.
- FedPer/FedRep: 일부 parameter는 서버가 평균하지 않고 client-local로 유지한다.
- Ditto: global model과 personalized local model을 동시에 유지한다.
- SCAFFOLD: server/client control variate state를 주고받는다.
- FedBN: batch-norm/statistics를 local로 둔다.
- Distillation/ensemble FL: weight averaging 대신 logits/teacher signal을 교환한다.

따라서 method를 하나의 `server_strategy` 문자열로만 보면 안 되고, 최소한 다음 축으로 나눠야 한다.

## 구현한 축

`federated_elsa_robotics/fl_method_registry.py`에 method spec을 확장했다.

- `server_strategy`: 서버 aggregation. 현재 기본은 `fedavg`.
- `client_update`: client가 local objective/state를 어떻게 쓰는지.
- `local_regularizer`: `none` 또는 `fedprox`.
- `parameter_scope`: 서버가 어떤 parameter surface를 aggregate하는지.
- `client_state`: client-local state가 필요한지.
- `local_parameter_keywords`: server aggregation에서 제외하고 client에 저장할 parameter name token.

새 preset:

- `fedper_head`
  - shared trainable body만 서버가 aggregate.
  - `policy_fc2`, `diffusion_head`, `multitoken_diffusion_head`, `gripper_head`는 client-local.
- `fedprox_fedper_head`
  - 위 구조에 FedProx proximal objective를 결합.
  - VolumeDP/diffusion에서 shared geometry/trunk는 global로 묶고 action/controller head는 local로 둘 때 쓴다.

## 코드 변경점

- `federated_elsa_robotics/parameter_surfaces.py`
  - aggregated parameter와 local-only parameter를 분리한다.
  - local-only tensor를 client별 파일로 저장/복원한다.
  - trainable manifest에 aggregated/local-only surface를 같이 기록한다.

- `federated_elsa_robotics/task.py`
  - `get_weights(agent, config)`와 `set_weights(agent, ..., config)`가 config-aware aggregated surface를 사용한다.
  - optimizer는 여전히 전체 trainable parameter를 학습한다.
  - FedProx proximal anchor는 aggregated shared parameter에만 걸리도록 바꿨다.

- `federated_elsa_robotics/client_app.py`
  - round 시작 시 서버 aggregated weights를 받고, local-only state가 있으면 복원한다.
  - local training 후 local-only state를 저장하고, 서버에는 aggregated weights만 반환한다.
  - metrics에 `loaded_local_tensors`, `saved_local_tensors`를 추가했다.

- `federated_elsa_robotics/server_app.py`
  - server 초기 parameter와 checkpoint load/save가 config-aware aggregated surface를 사용한다.

- `experiments/fl_dinov3_diffusion_lora4_jvdirect_fedprox_fedper_head.yaml`
  - FedProx + local diffusion/action head template.

- `federated_elsa_robotics/server_aggregation.py`
  - FedAvg, FedExp, FedNova, q-FedAvg, AFL, MaxFL의 server-side aggregation을 ndarray 단위로 구현한다.
  - AFL은 server-side client/domain lambda state를 유지한다.

- `scripts/eval_flower_personalized_local_state.py`
  - server checkpoint와 client-local state를 같이 로드해서 personalized offline validation을 수행한다.

## 현재 구현의 의미

이제 `fedavg`와 `fedprox`는 global-only FL이고, `fedper_head`/`fedprox_fedper_head`는 non-server-only FL이다. 서버는 shared body만 aggregate하고, client별 head는 checkpoint root 아래에 따로 유지된다.

local state 저장 위치:

```text
<checkpoint-root>/<task>/client_local_state/<run-tag>/partition_<id>_env_<env>.pt
```

주의:

- FedPer류 server checkpoint는 personalized local head를 포함하지 않는다. 즉 그대로 live global eval에 쓰면 local head가 초기 상태일 수 있다.
- 이 방법은 "global model 하나 제출" 실험과 분리해서 봐야 한다.
- 제대로 평가하려면 같은 client/env에서 local state를 복원한 personalized eval path가 필요하다.

## 아직 남은 축

SCAFFOLD:

- client별 control variate와 server control variate가 필요하다.
- parameter surface가 바뀌면 control variate shape도 같이 무효화해야 한다.
- 지금 구현한 `parameter_surfaces.py`와 client state path를 재사용하면 다음 단계로 넣을 수 있다.

Ditto:

- global update와 personalized local model update를 둘 다 수행해야 한다.
- 지금 local-only state 저장 구조를 확장해 personalized model 전체 또는 adapter-only state를 저장하면 된다.

q-FedAvg/MaxFL/FedNova:

- 이들은 여전히 server aggregation 축이 크지만, client telemetry가 필요하다.
- 현재 1차 구현은 들어갔다. `fednova`는 `local_steps`, `qfedavg`/`afl`/`maxfl`은 `pre_train_loss` 또는 `train_loss`, `fedexp`는 delta norm을 쓴다.
- `qfedavg`/`afl`/`maxfl`은 `metrics-probe-batches > 0`으로 pre-train loss를 넣는 쪽이 더 낫다.

## Runnable Smoke Matrix

| Preset | 서버 aggregation | client objective | local state | 기대 체크포인트/metric |
| --- | --- | --- | --- | --- |
| `fedavg` | FedAvg | standard | 없음 | global checkpoint만 저장 |
| `fedprox` | FedAvg | FedProx | 없음 | `prox_mu` drift baseline |
| `fedexp` | FedExp adaptive LR | standard | 없음 | `server_lr`, `fedexp_avg_delta_norm` |
| `fednova` | normalized delta | standard | 없음 | `fednova_effective_tau`, tau min/max |
| `qfedavg` | q-FFL style loss-aware update | standard | 없음 | `qffl_q`, `qffl_h_sum`, clipping flag |
| `afl` | AFL lambda-weighted update | standard | server lambda state | `afl_lambda_max`, known clients |
| `maxfl` | threshold/appeal-weighted update | standard | 없음 | `maxfl_weight_max`, threshold |
| `fedper_head` | shared body FedAvg | local heads train normally | client-local head tensors | local state `.pt` 생성 |
| `fedprox_fedper_head` | shared body FedAvg | FedProx on shared body | client-local head tensors | local state `.pt` + shared prox |

권장 순서:

1. `fedprox_fedper_head` 짧은 smoke run으로 parameter split과 local state save/load 확인.
2. personalized eval path 추가.
3. FedExp/q-FedAvg/MaxFL server strategy 추가.
4. SCAFFOLD control variate 추가.
