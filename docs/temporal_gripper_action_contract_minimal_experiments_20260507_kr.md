# Temporal Gripper Event Grounding + Action/Controller Contract 최소 검증 실험 계획 - 2026-05-07

목표는 `close_box`, `insert_onto_square_peg`, `scoop_with_spatula`의 same-env failure가 정말로
`temporal gripper event grounding`과 `action/controller contract` 때문인지 최소 비용으로 직접 검증하는 것이다.

전제:

- 현재 실행 중인 `ralph_fill4_power_moved_20260507` 큐는 건드리지 않는다.
- 새 실험은 현재 큐 완료 후 별도 output/log root에서만 실행한다.
- vision backbone 확장은 이번 묶음에서 제외한다. 같은 backbone과 비슷한 data split을 유지해 confound를 줄인다.

## 우선순위 요약

| Priority | ID | 종류 | 핵심 질문 |
| --- | --- | --- | --- |
| 1 | E1 | offline/eval-only | 현재 모델이 transition 주변 frame을 실제로 못 맞추는가? |
| 2 | E2 | eval-only | rollout-side gripper hysteresis만으로 SR이 오르는가? |
| 3 | E3 | eval-only | 같은 checkpoint를 다른 execution contract로 굴리면 결과가 바뀌는가? |
| 4 | E4 | small train | transition-weighted BCE가 non-slide SR을 0 이상으로 끌어올리는가? |
| 5 | E5 | small train | shorter chunk / shorter execution horizon이 timing 병목을 줄이는가? |
| 6 | E6 | conditional train | weighted BCE로도 안 되면 explicit event head가 필요한가? |

## E1. Transition-window diagnostic

가설:
현재 split gripper head는 전체 frame 평균 BCE는 버티지만, 실제 transition 전후 `+-K` frame에서는 예측 품질이 무너진다.

정확한 개입:

- validation loader에서 각 sample의 gripper target sequence를 읽고 transition index를 찾는다.
- 전체 frame metric과 transition-window metric을 분리해 계산한다.
- 가능하면 이미 완료된 checkpoint 3개만 본다.
- 추천 checkpoint:
  - `slide_jprel_w4_direct_grid16_eeaux_e50_s0`
  - 현재 큐 완료 후 best `close_*`
  - 현재 큐 완료 후 best `insert_*` 또는 `scoop_*`

대조군/베이스라인:

- same checkpoint의 global gripper accuracy / BCE
- same checkpoint의 non-transition frame metric

지표:

- `transition_window_bce`
- `transition_window_f1`
- `transition_frame_recall`
- `global_gripper_bce`
- `global_gripper_accuracy`
- sample당 `num_gripper_flips`

예상 결과:

- `slide`는 global과 transition metric 격차가 작거나 중간 수준이다.
- `close/insert/scoop`는 global metric 대비 transition-window F1이 크게 낮다.

실패 해석:

- transition-window metric도 이미 높으면 temporal grounding 자체보다 execution contract 또는 arm-target error가 더 큰 병목이다.
- task 전체에서 gripper flips가 거의 없거나 label이 이상하면 dataset labeling 먼저 점검해야 한다.

필요 코드 변경 범위:

- 신규 `scripts/analyze_gripper_transition_metrics.py`
- 필요 시 `elsa_learning_agent/agent.py`에 logits 반환용 얇은 helper 추가
- 필요 시 `scripts/train_same_env_bcpolicy_probe.py` 또는 eval script에서 checkpoint/config 로딩 재사용

성공/실패 판정 기준:

- 성공: non-slide task에서 `global_accuracy`는 높지만 `transition_window_f1`이 현저히 낮다는 분리 신호를 확보
- 실패: global/transition 모두 낮거나 모두 높아 temporal event 가설을 지지하지 못함

## E2. Rollout-only gripper hysteresis A/B

가설:
현재 `sigmoid >= 0.5` binary threshold가 1-2 frame oscillation을 만들고, RLBench gripper side effect 때문에 grasp/release를 망친다.

정확한 개입:

- 학습은 그대로 둔다.
- rollout 시 gripper decision만 아래 두 방식으로 비교한다.
  - baseline: 현재 `0.5` threshold
  - hysteresis: `close_on >= 0.65`, `open_on <= 0.35`, optional `min_hold_steps=2`
- 동일 checkpoint, 동일 env, 동일 eval episode 수로 A/B 평가한다.

대조군/베이스라인:

- 같은 checkpoint의 기본 threshold 평가
- 가능하면 task당 2개 checkpoint만 사용

지표:

- `sr`
- episode당 `num_gripper_state_changes`
- first successful grasp/release 이전 `flip_count`
- step 수

예상 결과:

- close/insert/scoop 중 최소 1개 task에서 SR이 오른다.
- flip count는 감소한다.

실패 해석:

- SR과 flip count가 모두 거의 안 바뀌면 rollout thresholding은 주 병목이 아니다.
- flip count만 줄고 SR이 안 오르면 timing 자체보다 arm pose alignment가 더 중요하다.

필요 코드 변경 범위:

- `elsa_learning_agent/agent.py` 또는 `elsa_learning_agent/live_rollout.py`
- 가능하면 config field 추가:
  - `dataset.gripper_eval_mode`
  - `dataset.gripper_close_threshold`
  - `dataset.gripper_open_threshold`
  - `dataset.gripper_min_hold_steps`
- `elsa_learning_agent/config_utils.py`
- `elsa_learning_agent/config_validation.py`
- `scripts/eval_flower_checkpoint_live.py`

성공/실패 판정 기준:

- 성공: non-slide task 중 하나라도 absolute SR `+0.10` 이상, 또는 flip count `30%+` 감소와 함께 SR 상승
- 실패: 모든 target task에서 SR 변화가 `+-0.05` 이내

## E3. Same-checkpoint controller-contract swap

가설:
policy target과 execution contract를 분리해 보면, 일부 task는 학습이 아니라 execution contract에서만 손해를 보고 있다.

정확한 개입:

- 같은 `jprel` checkpoint 하나를 저장된 weight 그대로 두고 평가 시 contract만 바꾼다.
- 추천 A/B:
  - `joint_position direct`
  - `joint_position -> joint_velocity servo`
- task별로 같은 checkpoint를 두 contract로만 평가한다.
- 가장 직접적인 대상:
  - `close_jprel_w4_*`
  - `insert_jprel_w4_*`
  - `scoop_jprel_w4_*`

대조군/베이스라인:

- 같은 checkpoint, 같은 seed, 같은 eval env, 같은 episode 수
- contract 외 모든 설정 고정

지표:

- `sr`
- `steps`
- servo contract일 때 `executed_steps`
- possible: per-episode mean joint delta residual

예상 결과:

- task마다 best contract가 갈린다.
- slide와 non-slide의 최적 contract가 같지 않을 수 있다.

실패 해석:

- 같은 checkpoint에서 contract swap이 거의 영향이 없으면 learned policy target/loss가 더 큰 병목이다.
- direct는 되고 servo만 안 되면 servo gain/clip/tolerance가 잘못된 confound일 수 있다.

필요 코드 변경 범위:

- `scripts/eval_flower_checkpoint_live.py`에 eval-time config override 추가
- `elsa_learning_agent/config_utils.py`
- 필요 시 `elsa_learning_agent/utils.py`의 adapter param logging 확장

성공/실패 판정 기준:

- 성공: 같은 checkpoint에서 contract 변경만으로 absolute SR `+0.10` 이상 차이 발생
- 실패: 모든 target task에서 차이가 미미해 controller contract 단독 효과가 약함

## E4. Transition-weighted BCE

가설:
non-slide 실패의 핵심은 sparse transition gradient dilution이며, gripper BCE를 transition 주변에 집중하면 same-env SR이 바로 살아난다.

정확한 개입:

- 기존 split gripper head는 유지한다.
- gripper label transition 전후 `+-8` frame에 weight `6x`를 준 weighted BCE를 추가한다.
- arm diffusion loss, vision backbone, dataset split, seed는 baseline과 동일하게 둔다.
- 최소 run만 한다:
  - `close`
  - 현재 큐에서 더 가능성 높은 non-slide 1개(`insert` 또는 `scoop`)
  - `slide` regression guard 1개

대조군/베이스라인:

- 현재/기존 동일 config run
- 예:
  - `close_jprel_w4_direct_grid16_eeaux_e50_s0` 또는 `close_jprel_w4_jvservo_grid16_eeaux_e50_s0`
  - `insert_jprel_w4_jvservo_grid16_eeaux_e50_s0`
  - `slide_jprel_w4_direct_grid16_eeaux_e50_s0`

지표:

- `sr`
- `offline_seen_env.mean_loss`
- E1에서 정의한 transition-window metrics
- training history 상 gripper-related auxiliary metric

예상 결과:

- non-slide 중 최소 1개에서 `SR >= 0.10`
- transition-window recall/F1 개선
- `slide`는 큰 regression 없이 유지

실패 해석:

- transition metric만 좋아지고 SR이 안 오르면 execution-side hysteresis/contract가 남은 병목이다.
- metric도 안 좋아지면 label window, weight scale, 또는 chunk formulation이 맞지 않는다.

필요 코드 변경 범위:

- `elsa_learning_agent/dataset/dataset_loader.py`에 transition mask 산출
- `elsa_learning_agent/agent.py`에 weighted BCE 지원
- `elsa_learning_agent/config_utils.py`
- `elsa_learning_agent/config_validation.py`
- `scripts/train_same_env_bcpolicy_probe.py` result logging 확장
- 신규 experiment yaml 3개 내외

성공/실패 판정 기준:

- 성공: non-slide 최소 1개 `SR >= 0.10` 그리고 `slide` absolute drop `< 0.10`
- 실패: non-slide 모두 `SR < 0.05`

## E5. Chunk/horizon contraction

가설:
action chunk가 arm smoothing에는 도움이 되지만 gripper event timing에는 너무 둔하다. 특히 `w4/exec2`가 non-slide에서 transition을 늦춘다.

정확한 개입:

- E4에서 가장 잘 나온 non-slide task 하나만 고른다.
- 같은 loss와 contract에서 아래 둘만 비교한다.
  - baseline: current best `w4`
  - intervention: `w2` and `execute_steps=1`
- policy target family와 backbone은 고정한다.

대조군/베이스라인:

- E4 best run
- 현재 큐에서 나온 `w2` 결과가 있으면 재사용하고, 없을 때만 train

지표:

- `sr`
- transition-window F1
- episode step count
- rollout 중 gripper flip timing variance

예상 결과:

- shorter horizon이 non-slide timing을 개선하면 SR이 오른다.

실패 해석:

- improvement가 없으면 문제는 chunk length보다 label/loss 또는 contract다.
- slide만 좋아지고 non-slide가 그대로면 horizon 문제는 task-specific이다.

필요 코드 변경 범위:

- 대부분 config/YAML만 변경
- 추가 코드가 필요하면 `scripts/eval_flower_checkpoint_live.py`의 execute-step override 정도

성공/실패 판정 기준:

- 성공: chosen non-slide task에서 absolute SR `+0.10` 이상
- 실패: SR 변화가 작고 transition metric도 개선 없음

## E6. Explicit event head (조건부)

가설:
binary open/close supervision만으로는 희소 event를 분리하기 어려워, explicit phase/event classification이 필요하다.

정확한 개입:

- gripper output을 `{hold_open, close, hold_closed, open}` 4-class head로 바꾼다.
- execution에서는 current state와 class를 조합해 binary command를 생성한다.
- label은 transition 주변을 넓혀 smoothed event class로 만든다.
- E4/E5가 실패했을 때만 1개 non-slide task + slide guard로 작게 검증한다.

대조군/베이스라인:

- E4 best weighted-BCE model

지표:

- `sr`
- event-class macro F1
- transition recall
- rollout flip count

예상 결과:

- weighted BCE보다 event localization이 더 좋아진다.

실패 해석:

- 이것도 실패하면 gripper event 자체보다 pose grounding 또는 interaction point supervision 부족 가능성이 커진다.

필요 코드 변경 범위:

- `elsa_learning_agent/dataset/dataset_loader.py`
- `elsa_learning_agent/agent.py`
- `elsa_learning_agent/config_utils.py`
- `elsa_learning_agent/config_validation.py`
- `scripts/train_same_env_bcpolicy_probe.py`
- 신규 experiment yaml

성공/실패 판정 기준:

- 성공: chosen non-slide task에서 E4 대비 absolute SR `+0.10` 이상
- 실패: metric 개선 없이 복잡도만 증가

## 최소 실행 순서

1. 현재 큐 완료 후 결과 집계로 non-slide baseline 1st/2nd candidate를 정한다.
2. `E1`을 먼저 돌려 temporal event hypothesis가 실제로 보이는지 확인한다.
3. training 없이 답이 나오는 `E2`, `E3`를 먼저 실행한다.
4. 여전히 non-slide가 0이면 `E4`를 3-run 묶음으로 실행한다.
5. `E4`가 부분 성공이면 `E5`는 best non-slide task 1개만 본다.
6. `E4/E5`가 모두 실패하고 `E1/E2`가 timing sensitivity를 지지할 때만 `E6`로 간다.

이 순서는 "가설을 직접 테스트하는 값싼 실험"에서 "작은 학습 개입"으로만 확장하므로 불필요한 queue 증식을 막는다.

## 필요한 코드 변경 범위 요약

낮은 범위:

- `scripts/eval_flower_checkpoint_live.py`
- 신규 `scripts/analyze_gripper_transition_metrics.py`
- experiment yaml 몇 개

중간 범위:

- `elsa_learning_agent/live_rollout.py`
- `elsa_learning_agent/utils.py`
- `elsa_learning_agent/config_utils.py`
- `elsa_learning_agent/config_validation.py`

학습 로직 변경 범위:

- `elsa_learning_agent/dataset/dataset_loader.py`
- `elsa_learning_agent/agent.py`
- `scripts/train_same_env_bcpolicy_probe.py`

원칙:

- `E1-E3`는 queue 안전성이 높고 기존 checkpoint 재사용이 가능하다.
- 실제 학습 코드 변경은 `E4`부터 시작한다.
- `E6`는 가장 침습적이므로 반드시 앞선 실험 실패 뒤에만 연다.

## 성공/실패 총괄 게이트

다음 단계로 넘어가는 최소 성공 조건:

- `E2` 또는 `E3`에서 eval-only 개입만으로 non-slide absolute SR `+0.10` 이상
- 또는 `E4`에서 non-slide 최소 1개 `SR >= 0.10`

temporal grounding 가설 지지 조건:

- `E1`에서 transition-window metric이 global metric보다 명확히 나쁨
- `E4` 또는 `E6`에서 transition metric과 SR이 함께 개선

controller contract 가설 지지 조건:

- `E3`에서 same checkpoint contract swap만으로 SR 차이가 유의미함
- `E2`에서 hysteresis가 SR을 올리고 flip count를 줄임

가설 반박 조건:

- `E1-E3`에서 timing/contract 민감도가 작고
- `E4-E6`에서도 non-slide 개선이 없으면
- 병목은 gripper timing보다 interaction point, pose target, 또는 vision grounding일 가능성이 높다.

## Run naming / logging 원칙

현재 큐 결과와 섞이지 않게 아래 규칙을 고정한다.

1. 기존 queue root를 재사용하지 않는다.
   - 금지: `results/action_ablation_20260504`, `results/overnight_queue`, `results/recommended_followups_20260504`
   - 권장:
     - results root: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/tgac_20260507`
     - ckpt root: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints/tgac_20260507`
     - log root: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/tgac_20260507`

2. 모든 run name에 고정 family tag를 넣는다.
   - 형식:
     - `{task}_{basepolicy}_tgac0607_{expid}_e{epochs}_s{seed}`
   - 예:
     - `close_jprel_w4_direct_grid16_eeaux_tgac0607_evalhy_e0_s0`
     - `insert_jprel_w4_jvservo_grid16_eeaux_tgac0607_gripw6xk8_e50_s0`

3. `expid`는 가설을 직접 드러내야 한다.
   - `diagtrans`
   - `evalhy`
   - `evalcontract`
   - `gripw6xk8`
   - `w2x1`
   - `evt4`

4. eval-only 결과도 `result.json` 외에 `eval_spec.json`을 남긴다.
   - 포함 필드:
     - `hypothesis_id`
     - `baseline_run_name`
     - `checkpoint_path`
     - `config_override`
     - `git_commit`
     - `queue_family=tgac_20260507`

5. training result에는 baseline reference를 같이 남긴다.
   - `baseline_run_name`
   - `baseline_sr`
   - `experiment_family`
   - `transition_window`
   - `transition_weight`
   - `gripper_eval_mode`

6. 로그 파일명도 run name과 1:1로 맞춘다.
   - `{log_root}/{run_name}.log`
   - queue summary는 `{log_root}/_tgac_status.log`

7. 기존 큐 skip logic와 충돌하지 않게 run name을 절대 재사용하지 않는다.

8. 결과 집계 표에서도 queue/family를 column으로 분리한다.
   - 최소 column:
     - `family`
     - `task`
     - `run_name`
     - `baseline_run_name`
     - `action_representation`
     - `execution_action_interface`
     - `execution_action_adapter`
     - `transition_weight`
     - `sr`

## 권장 첫 묶음

현재 큐 완료 직후 가장 먼저 할 최소 묶음은 아래다.

- `E1` on 3 checkpoints
- `E2` on best `close` and best `insert/scoop`
- `E3` on same two checkpoints
- `E4` only 3 train runs:
  - `close`
  - best non-slide candidate 1개
  - `slide` regression guard

이 4개만으로도
`temporal grounding이 진짜 병목인가`,
`contract가 rollout에서 실제로 중요하게 작동하는가`,
`학습 loss를 바꾸면 non-slide가 살아나는가`
를 충분히 판별할 수 있다.

## 2026-05-07 코드 준비 상태

현재 큐를 새로 띄우지 않고, 다음 실험을 켤 수 있는 코드만 준비했다. 기존 config의 기본 동작은 유지된다.

구현된 것:

- `E1`: `scripts/analyze_gripper_transition_metrics.py`
  - checkpoint의 global gripper metric과 transition-window metric을 분리한다.
  - `--transition-window 8`로 expert gripper transition 주변 frame만 따로 집계한다.
- `E2`: rollout-only gripper hysteresis
  - `dataset.gripper_eval_mode: hysteresis` 또는 eval script의 `--gripper-eval-mode hysteresis`로 켠다.
  - 기본값은 `threshold`라 기존 실험은 그대로다.
- `E3`: eval-time controller contract override
  - `scripts/eval_flower_checkpoint_live.py`에 action pipeline, execution interface/adapter, servo gain/clip/steps override를 추가했다.
- `E4`: transition-weighted BCE
  - `model.gripper_transition_window`와 `model.gripper_transition_weight`로 켠다.
  - split gripper head가 켜진 config에서만 `gripper_target_weight`가 dataloader batch에 추가된다.
- 결과 메타데이터:
  - `result.json`에 `git_sha`, `boot_id`, `hostname`, `config_sha256`, gripper transition/eval 설정을 남긴다.
- 집계:
  - `scripts/aggregate_sameenv_sweep_results.py`와 `scripts/wave_summary.py`에 transition weight/window, gripper eval mode, family column을 추가했다.
- preflight:
  - `scripts/tgac_preflight_20260507.sh`는 현재 queue를 시작/중단하지 않고 TGAC 코드와 config를 검증한다.
- baseline candidate scan:
  - `scripts/tgac_collect_baseline_candidates_20260507.py`는 artifact results에서 E1-E3 후보 checkpoint/config를 task별로 모은다.
  - 현재 snapshot은 `docs/tgac_baseline_candidates_20260507.json`에 저장했다.

준비된 E4 YAML:

- `experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux_gripw6xk8.yaml`
- `experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_jvservo_grid16_eeaux_gripw6xk8.yaml`
- `experiments/tgac_20260507_plan.json`

현재 큐가 끝난 뒤 권장 실행 root:

- results: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/tgac_20260507`
- checkpoints: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints/tgac_20260507`
- logs: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/tgac_20260507`
