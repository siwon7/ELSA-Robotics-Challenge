# Harness-100 Codex Handoff for ELSA/Ralph/FLAME - 2026-05-07

이 문서는 다른 Codex 또는 Claude Code 인스턴스가 `/home/cvlab-dgx/siwon/harness-100`를 이용해 ELSA/Ralph/FLAME 연구를 이어받기 위한 실행 지침이다. 지금까지의 코드 변경, 실험 결과, 전원 장애 기록, 큐 스케줄링 상태, 다음 실험 우선순위를 하나로 묶었다.

## 0. 결론 먼저

- 목표는 FL 전에 `same env` 4개 task 성능을 먼저 올리는 것이다.
- 현재 가장 강한 신호는 `slide_block_to_target`뿐이다. 최고 SR은 `jprel_w4_direct_grid16_eeaux`의 `0.85`다.
- `close_box`, `insert_onto_square_peg`, `scoop_with_spatula`는 replay ceiling은 높지만 learned policy가 거의 못 풀고 있다.
- 따라서 다음 핵심은 backbone/VolumeDP를 더 키우는 것이 아니라 `연속 arm action + 희소 gripper event timing`을 제대로 학습시키는 것이다.
- 현재 4GPU 큐는 전원 위치 변경 후 `ralph_fill4_power_moved_20260507` 세션에서 실행 중이다. 중복 스케줄러를 띄우지 말고 먼저 이 큐를 관찰해야 한다.
- 전원 장애는 Python/OOM보다 PSU/UPS/PDU/outlet/board/power spike 쪽 가능성이 높다. root 권한 없이 GPU power cap은 적용하지 못했다.

## 1. 경로

| 항목 | 경로 |
| --- | --- |
| 메인 레포 | `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge` |
| harness-100 | `/home/cvlab-dgx/siwon/harness-100` |
| artifact root | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts` |
| 현재 4GPU log root | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507` |
| 현재 master log | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507/fill3_master.log` |
| 현재 scheduler | `scripts/run_cpu_limited_fill3_queues_20260506.sh` |

## 2. Harness-100 사용법

추천 하네스는 `ko/31-ml-experiment`다.

```bash
cd /home/cvlab-dgx/siwon/harness-100
./scripts/harness.sh info ko 31
./scripts/harness.sh apply ko 31-ml-experiment /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge
cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge
claude
```

주의할 점:

- `LOCAL_SETUP_KO.md`에는 예전 `/mnt/siwon/harness-100` 경로가 적혀 있지만, 실제 경로는 `/home/cvlab-dgx/siwon/harness-100`다.
- 하네스 적용 시 기존 `.claude`는 자동으로 `.claude.backup.TIMESTAMP`로 백업된다.
- 이미 다른 실험 에이전트 설정이 있으면 하네스 적용 전 `.claude` 내용을 확인한다.
- `31-ml-experiment`는 `_workspace/`에 산출물을 만든다.
- IC613 FL smoke는 현재 `ralph_fill4_power_moved_20260507` 큐를 건드리지 말고 CPU-only 고유 artifact root로 돌리거나, 해당 큐가 idle 된 뒤 GPU에 올린다.

`31-ml-experiment`의 역할 매핑:

| Harness agent | 이 연구에서 맡길 일 |
| --- | --- |
| `data-engineer` | FLAME/Ralph trajectory, action 변환, gripper transition 통계 검증 |
| `model-designer` | gripper event head, transition-weighted BCE, action interface 설계 |
| `training-manager` | tmux/GPU 큐, 전원 안정성, result skip, 재현성 관리 |
| `evaluation-analyst` | SR 집계, task별 failure mode, replay ceiling 대비 분석 |
| `experiment-reviewer` | confound 방지, 논문 주장 가능성 검토 |

## 3. 현재 연구 목표

현재 목표는 다음 순서로 고정한다.

1. `same env` 4개 task에서 learned policy가 최소한 non-zero signal을 내도록 만든다.
2. 성공 신호가 있는 action/vision 조합을 고정한다.
3. 그 조합으로 FLAME federated learning baseline을 돌린다.
4. FL에서 FedAvg, FedProx, SCAFFOLD 또는 LoRA-only aggregation을 비교한다.
5. 충분한 결과가 나오면 `docs/proposal.pdf`와 논문 초안에 반영한다.

대상 task:

| Short name | Task |
| --- | --- |
| `close` | `close_box` |
| `slide` | `slide_block_to_target` |
| `insert` | `insert_onto_square_peg` |
| `scoop` | `scoop_with_spatula` |

## 4. 현재 핵심 발견

### 4.1 Replay ceiling

Replay는 actuator/action interface가 본질적으로 불가능하지 않다는 것을 보여준다.

| Action replay mode | close | slide | insert | scoop |
| --- | ---: | ---: | ---: | ---: |
| JV direct | 1.0 | 1.0 | 0.0 | 0.2 |
| JP direct | 1.0 | 1.0 | 0.8 | 0.9 |
| JP to JV servo | 0.9 | 1.0 | 0.9 | 0.9 |

해석:

- low-level simulator/actuator injection만으로는 실패를 설명하기 어렵다.
- learned policy가 action target, gripper timing, visual grounding을 못 맞추는 쪽이 병목이다.
- insert/scoop은 JP 또는 JP-to-JV servo ceiling이 높으므로 무조건 keyframe으로 갈 필요는 없다.

### 4.2 현재 결과 요약

| Queue | Task | Run | SR |
| --- | --- | --- | ---: |
| `relative_action_20260504` | `slide` | `slide_jprel_w4_direct_grid16_eeaux_e50_s0` | 0.85 |
| `overnight_queue` | `slide` | `slide_volumedp_w4_eeaux_e50` | 0.60 |
| `action_ablation_20260504` | `slide` | `slide_jprel_w4_jvservo_grid16_eeaux_e50_s0` | 0.55 |
| `overnight_queue` | `slide` | `slide_volumedp_w4_e100` | 0.50 |
| `overnight_queue` | `slide` | `slide_volumedp_w4_film_eeaux_e50` | 0.30 |
| `overnight_queue` | `close` | `close_volumedp_w4_proprio_film_e50` | 0.20 |
| `overnight_queue` | `close` | `close_volumedp_w4_eeaux_5env_e10` | 0.16 mean |

부정 결과:

- `paperfaithful_20260504`는 4개 task 모두 완료됐지만 SR은 전부 `0.0`이다.
- `relative_action_20260504`는 slide만 `0.85`이고 close/insert/scoop은 `0.0`이다.
- insert/scoop은 아직 안정적인 성공 신호가 거의 없다.

### 4.3 Paper-faithful VolumeDP 해석

`paperfaithful_20260504`가 same env도 못 푼다고 해서 논문 아이디어 자체가 틀렸다고 결론내리면 안 된다. 현재 구현은 camera metadata, interaction-region target, gripper-transition auxiliary 등이 원 논문의 중요한 구성과 다를 수 있고, FLAME task의 gripper event가 더 희소하다.

하지만 실험 우선순위 관점에서는 명확하다. 지금은 VolumeDP fidelity를 더 높이는 것보다 learned policy의 action/gripper bottleneck을 먼저 해결해야 한다.

## 5. 새로운 관점

이 연구를 `좋은 vision backbone 찾기`로 보면 진행이 느려진다. 현재 더 생산적인 관점은 다음이다.

### 5.1 Temporal event grounding problem

Close/insert/scoop 실패는 arm trajectory 평균 오차보다 `언제 gripper를 닫고 여는가`가 더 치명적일 가능성이 높다. gripper transition은 episode당 보통 1-2회뿐이라 일반 BCE 또는 chunk BCE에서 gradient가 희석된다.

따라서 다음 실험은 다음 질문에 답해야 한다.

- 모델이 gripper transition frame 주변을 구분하는가?
- gripper state가 한두 frame 흔들릴 때 simulator side effect 때문에 grasp/release가 망가지지 않는가?
- action chunk 길이가 arm smoothing에는 좋지만 gripper timing에는 너무 둔하지 않은가?

### 5.2 Controller contract separation

Action 연구는 하나의 이름으로 뭉뚱그리면 안 된다. 논문/실험 표에는 최소한 두 층을 분리해서 기록해야 한다.

| 층 | 예시 | 의미 |
| --- | --- | --- |
| Policy target | `jprel`, `jpabs`, EE pose, keyframe | 네트워크가 예측하는 supervision target |
| Execution contract | direct joint position, JV direct, JP-to-JV servo, hysteresis gripper | rollout에서 simulator에 주는 control interface |

Replay ceiling이 높고 learned policy가 낮으면 execution contract를 계속 바꾸기보다 policy target/loss/event supervision을 봐야 한다.

### 5.3 Vision은 두 문제로 분리

| 문제 | 적합한 방향 |
| --- | --- |
| texture/light/domain robustness | frozen DINOv3, LoRA, Depth Anything |
| viewpoint/camera geometry | VolumeDP-lite, camera-aligned voxel/token, extrinsics |

same env에서 non-slide task가 0이면 viewpoint robustness가 주 병목이라고 보기 어렵다. 지금은 action/gripper를 먼저 고치고, 이후 multi-env/FL로 넘어갈 때 VolumeDP-lite를 다시 강화한다.

### 5.4 Federated learning은 성능 증폭기가 아니다

FL은 client drift와 aggregation을 다루는 방법이지, same-env single-client가 못 푸는 task를 자동으로 풀어주는 방법이 아니다. 따라서 FL은 다음 조건 후에 붙인다.

- 최소 2개 이상의 task에서 same-env SR이 non-zero로 안정적으로 나온다.
- slide 외 task에서 action/gripper 변경으로 개선 신호가 나온다.
- 단일 환경 rollout failure mode를 설명할 수 있다.

## 6. 현재 실행 중인 큐

2026-05-07 KST 기준 현재 실행 중인 세션:

```bash
tmux attach -t ralph_fill4_power_moved_20260507
tmux attach -t power_watch_20260507_moved
```

상태 확인:

```bash
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507/fill3_master.log
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits
```

실행 설정:

| 항목 | 값 |
| --- | --- |
| tmux session | `ralph_fill4_power_moved_20260507` |
| power monitor session | `power_watch_20260507_moved` |
| scheduler | `scripts/run_cpu_limited_fill3_queues_20260506.sh` |
| `MAX_PARALLEL` | `4` |
| `BATCH_SIZE` | `16` |
| `ELSA_CPU_CORES_PER_GPU` | `4` |
| `ELSA_CPU_THREADS_PER_JOB` | `1` |
| `ELSA_DATALOADER_WORKERS` | `1` |
| `NUM_WORKERS` | `1` |
| `POLL_SEC` | `60` |

마지막 확인 상태:

- 시작 시각: 2026-05-07 05:58:54 KST
- master log: `active_workers=4 launched=4/16`
- 현재 의미: 16개 target 중 4개 worker가 실행 중이고, result가 이미 있는 run은 스킵된다.
- 예상 완료 시간: 서버가 꺼지지 않으면 대략 24-36시간.
- 주의: 중단된 training은 checkpoint resume이 아니라, `result.json`이 없으면 처음부터 다시 시작하는 것으로 취급한다.

첫 batch에서 관찰된 run:

| GPU | Run |
| ---: | --- |
| 0 | `slide_jprel_w2_direct_grid16_eeaux_e50_s0` |
| 1 | `close_jprel_w4_jvservo_grid16_eeaux_e50_s0` |
| 2 | `insert_jprel_w4_jvservo_grid16_eeaux_e50_s0` |
| 3 | `scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0` |

절대 하지 말 것:

- 같은 큐를 다른 tmux에서 중복으로 띄우지 않는다.
- `pkill CoppeliaSim` 같은 broad kill을 쓰지 않는다. 병렬 worker를 같이 죽일 수 있다.
- 전원 안정성 확인 없이 무작정 batch/GPU 수를 늘리지 않는다.

## 7. 남은 큐와 목적

현재 scheduler가 채우는 계열:

| Queue | 목적 |
| --- | --- |
| `action_ablation_20260504` | `jprel`, `jvservo`, `direct`, `w2/w4` action 비교 |
| `jpabs_seedsweep_20260504` | absolute joint position 계열과 seed 민감도 확인 |
| `overnight_queue` | 기존 VolumeDP/EE auxiliary 미완료분 마무리 |
| `recommended_followups_20260504` | slide/close 중심 follow-up |

기록된 미완료 또는 재실행 대상:

| Queue | Run |
| --- | --- |
| action | `slide_jprel_w2_direct_grid16_eeaux_e50_s0` |
| action | `close_jprel_w4_jvservo_grid16_eeaux_e50_s0` |
| action | `close_jprel_w2_direct_grid16_eeaux_e50_s0` |
| action | `insert_jprel_w4_jvservo_grid16_eeaux_e50_s0` |
| action | `insert_jprel_w2_direct_grid16_eeaux_e50_s0` |
| action | `scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0` |
| action | `scoop_jprel_w2_direct_grid16_eeaux_e50_s0` |
| overnight | `insert_volumedp_w4_eeaux_e50` |
| recommended | `slide_volumedp_w4_eeaux_5env_e10_s0` |
| recommended | `close_volumedp_w4_eeaux_e100_s0` |
| jpabs | `slide_w2_jvdirect_grid16_e50_s1` |
| jpabs | `slide_jpabs_w2_grid16_e50_s0` |
| jpabs | `slide_w2_jvdirect_grid16_e50_s2` |
| jpabs | `insert_jpabs_w2_grid16_e50_s0` |
| jpabs | `scoop_jpabs_w2_grid16_e50_s0` |

`close_volumedp_w4_eeaux_e100_s0`는 conditional run이다. 기준 run SR이 낮아서 script상 skip될 가능성이 있다.

## 8. 다음 Codex가 바로 할 일

### 8.1 첫 10분

```bash
cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge
git status --short --branch
tmux ls
tail -n 80 /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507/fill3_master.log
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits
```

확인해야 할 것:

- `ralph_fill4_power_moved_20260507`가 살아 있는가.
- `active_workers`가 0이 아니거나, 16개 launch가 끝났는가.
- 서버가 hard reset 되었는가.
- 새 `result.json`이 쌓였는가.

### 8.2 결과 집계

`result.json`이 쌓이면 queue별 SR을 표로 만든다.

```bash
find /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results \
  -path '*/result.json' -print | sort
```

권장 산출물:

- `docs/harness100_codex_handoff_20260507_kr.md` 업데이트
- `docs/ralph_fill4_restart_20260507_kr.md` 업데이트
- `docs/experiment_queue_record_20260504_kr.md` 업데이트
- 검증된 핵심 결과만 `docs/proposal.pdf`에 반영

### 8.3 큐 완료 후 판단 기준

| 관찰 | 다음 행동 |
| --- | --- |
| `jprel_w2_direct`가 slide에서 w4보다 좋음 | shorter chunk가 gripper/event에도 유리한지 close/insert/scoop에 확장 |
| `jprel_w4_jvservo`가 close/insert/scoop에서 non-zero | JP-to-JV servo 계열을 task별 기본 action 후보로 승격 |
| 모든 non-slide가 계속 0 | gripper transition loss/head를 즉시 구현 |
| insert/scoop만 계속 0 | task-specific interaction point 또는 phase supervision 추가 |
| VolumeDP만 개선 | vision/token 쪽으로 2차 ablation 확장 |

## 9. 다음 실험 우선순위

현재 큐가 끝나기 전에는 새 큐를 중복으로 넣지 않는다. 끝난 뒤에는 아래 순서로 진행한다.

### P0. Gripper transition-weighted BCE

가장 안전하고 기대값이 높은 실험이다. action space는 유지하고 gripper supervision만 강화한다.

아이디어:

- gripper state change frame을 찾는다.
- transition 전후 `+-8` frame 또는 `+-10` frame의 gripper BCE weight를 4-8배로 준다.
- arm diffusion loss는 그대로 둔다.
- 우선 `jprel_w4_direct_grid16_eeaux`와 `jprel_w4_jvservo_grid16_eeaux`에 붙인다.

실험:

| Priority | Task/action |
| --- | --- |
| 1 | `close_jprel_w4_direct_grid16_eeaux_gripw_e50_s0` |
| 2 | `insert_jprel_w4_jvservo_grid16_eeaux_gripw_e50_s0` |
| 3 | `scoop_jprel_w4_jvservo_grid16_eeaux_gripw_e50_s0` |
| 4 | `slide_jprel_w4_direct_grid16_eeaux_gripw_e50_s0` as regression guard |

성공 기준:

- close/insert/scoop 중 하나라도 SR `0.10` 이상이면 다음 단계로 확장한다.
- slide가 크게 무너지면 gripper loss weight가 arm policy를 방해하는지 확인한다.

### P1. Gripper hysteresis at rollout

학습을 거의 건드리지 않는 execution-side 안정화다.

아이디어:

- 단순 threshold `0.5` 대신 현재 gripper state 기반 hysteresis를 둔다.
- open to close는 `p_close > 0.65`, close to open은 `p_open > 0.65` 같은 식으로 흔들림을 줄인다.
- RLBench gripper는 grasp/release side effect가 크기 때문에 한두 frame oscillation이 치명적일 수 있다.

### P2. Gripper phase/event head

loss만으로 부족하면 직접 event를 예측한다.

아이디어:

- gripper binary 대신 또는 추가로 `{hold_open, close, hold_closed, open}` 4-class를 예측한다.
- execution에서는 current state와 class를 조합해 binary command를 만든다.
- event class는 transition frame 주변을 넓혀 label smoothing을 적용한다.

### P3. Task-conditioned action contract

모든 task에 하나의 action interface를 강제하지 않는다.

초기 후보:

| Task | Candidate |
| --- | --- |
| slide | `jprel_w4_direct_grid16_eeaux` |
| close | `jprel_w4_direct` plus grip transition or `jprel_w4_jvservo` |
| insert | `jpabs_w2` or `jprel_w4_jvservo` plus grip transition |
| scoop | `jpabs_w2` or `jprel_w4_jvservo` plus grip transition |

논문에는 `task-aware controller contract` 또는 `action contract ablation`으로 정리할 수 있다.

### P4. VolumeDP-lite after action fix

VolumeDP-lite는 action/gripper가 최소한 non-zero가 된 뒤 적용한다.

우선순위:

- DINO frozen + LoRA
- Depth Anything feature
- camera-aligned voxel/token
- EE-mask auxiliary
- gripper-transition interaction-region auxiliary

### P5. Federated learning

FL은 same-env에서 action/gripper가 고정된 뒤 붙인다.

우선순위:

| Priority | Method |
| --- | --- |
| 1 | FedAvg with frozen DINO and LoRA/head aggregation |
| 2 | FedAvg plus local FedProx penalty as baseline |
| 3 | SCAFFOLD for client drift correction |
| 4 | FedBN/personalized head only if evaluation protocol allows personalization |

초기 설정:

- `rounds=20`
- `local_epochs=5`
- `fraction_fit=0.05`
- chosen action: current queue 결과 후 결정
- CPU smoke first, then GPU full

## 10. 전원 장애와 스케줄링 정책

관찰된 사실:

- 여러 번 정상 shutdown 없이 hard reset이 발생했다.
- OOM, kernel panic, NVRM/Xid, thermal, MCE, pstore artifact는 뚜렷하지 않았다.
- GPU temperature는 낮았다.
- 4GPU 학습에서 합산 GPU power가 1kW 근처까지 관측됐고 순간 peak는 더 높을 수 있다.
- 2GPU에서도 한 번 crash가 있었다.
- 사용자가 전원 위치를 옮긴 뒤 현재 4GPU 큐가 이전보다 오래 살아 있다.

해석:

- 원인은 Python 코드보다는 power delivery 가능성이 높다.
- PSU/UPS/PDU/outlet/board/shared line 문제를 우선 의심한다.
- root 없이 `nvidia-smi -pl` power cap을 걸 수 없다.

crash 후 확인:

```bash
last -x | head -40
journalctl --list-boots
tail -n 80 /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507/fill3_master.log
tail -n 80 /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/power_health_20260506/power_health.log
```

root/admin이 있으면:

```bash
sudo bash scripts/apply_gpu_power_limit.sh 150
nvidia-smi -q -d POWER | grep -E 'GPU [0-9]|Power Limit|Power Draw'
```

운영 정책:

- 현재 4GPU 큐가 계속 살아 있으면 그대로 둔다.
- 또 hard reset되면 무조건 결과 파일 기준으로 재개한다.
- 전원 안정성이 의심되면 `MAX_PARALLEL=1` 또는 `2`로 낮춘다.
- 연구 실험 설계와 전원 안정성 실험을 섞지 않는다. crash가 잦은 날의 SR 결론은 신뢰하지 않는다.

## 11. 문서와 코드 근거

읽어야 할 주요 문서:

| 문서 | 내용 |
| --- | --- |
| `docs/research_idea_summary_kr.md` | 전체 연구 아이디어, FLAME, DINO/LoRA/Depth/VolumeDP/FL 요약 |
| `docs/replay_and_sameenv_handoff_kr.md` | replay ceiling과 same-env 병목 |
| `docs/action_space_gripper_continuous_survey_20260505_kr.md` | RLBench/PerAct/RVT/ACT/Diffusion Policy/DP3 action/gripper 조사 |
| `docs/action_space_sota_findings_20260504_kr.md` | action space SOTA와 ablation 방향 |
| `docs/same_env_vision_action_suite_kr.md` | same-env vision/action 실험 계획 |
| `docs/volumedp_federated_design_kr.md` | VolumeDP-lite와 federated 설계 |
| `docs/federated_method_plan_kr.md` | FedAvg/FedProx/SCAFFOLD 계획 |
| `docs/fl_experiment_roadmap_kr.md` | FL 실험 로드맵 |
| `docs/dgx_power_queue_recovery_20260506_kr.md` | 전원 장애와 conservative recovery 정책 |
| `docs/ralph_fill4_restart_20260507_kr.md` | 현재 4GPU 재시작 기록 |

중요 스크립트:

| 스크립트 | 용도 |
| --- | --- |
| `scripts/run_cpu_limited_fill3_queues_20260506.sh` | 여러 queue를 GPU slot에 채우는 현재 scheduler |
| `scripts/start_action_ablation_queue_tmux.sh` | action ablation worker |
| `scripts/start_jpabs_seedsweep_queue_tmux.sh` | JPABS seed sweep worker |
| `scripts/start_overnight_queue_pending_tmux.sh` | overnight 미완료분 |
| `scripts/start_recommended_followup_queue_tmux.sh` | 추천 follow-up |
| `scripts/monitor_power_health_20260506.sh` | power/health monitor |
| `scripts/check_power_health_20260506.sh` | reboot/power health 확인 |
| `scripts/apply_gpu_power_limit.sh` | root 권한 power cap 적용 |

## 12. Git 상태와 주의

현재 브랜치:

```bash
git status --short --branch
```

기록 시점 상태:

- branch: `ceiling-sweep-2026-04-30`
- remote 대비 local ahead 상태였다.
- 최근 기록 커밋:
  - `b00517f Document Ralph fill4 restart queue`
  - `f5a3366 Document Ralph fill2 fallback`
  - `ac4e8ae Document Ralph fill4 retry after power move`
- `docs/proposal_overleaf.zip`는 untracked artifact로 보였고, 사용자가 요청하지 않으면 건드리지 않는다.

주의:

- 사용자 변경을 revert하지 않는다.
- 실험 결과 문서는 커밋해도 되지만, 생성 artifact zip은 명시 요청 전에는 제외한다.
- proposal PDF 업데이트는 검증된 결과가 나온 뒤에만 한다.

## 13. 다른 Codex에게 줄 실행 프롬프트

다음 프롬프트를 harness 적용 후 Claude/Codex에 전달하면 된다.

```text
You are continuing the ELSA/Ralph/FLAME robotics experiment in /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge using harness-100 ko/31-ml-experiment.

Read docs/harness100_codex_handoff_20260507_kr.md first, then check the active tmux session ralph_fill4_power_moved_20260507 and the master log under /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507/fill3_master.log.

Do not start a duplicate scheduler. First monitor the current 4GPU queue, aggregate any new result.json files, and update the experiment documents. If the queue finishes, decide the next experiments using the priority order in the handoff: gripper transition-weighted BCE, gripper hysteresis, gripper event head, then VolumeDP-lite and FL.

Treat the main research bottleneck as temporal gripper event grounding plus action/controller contract, not just vision backbone selection. Preserve user changes and avoid broad process kills.
```

## 14. Success criteria for the next 24-48 hours

가장 좋은 1차 산출물:

- 현재 큐가 crash 없이 끝난다.
- action ablation 결과표가 완성된다.
- close/insert/scoop 중 하나라도 non-zero SR이 나온다.
- non-slide가 계속 0이면 gripper transition-weighted BCE 구현과 작은 큐가 준비된다.

논문 관점의 1차 성공:

- `slide`는 이미 `0.85` signal이 있으므로, 이 결과를 baseline success로 둔다.
- `close/insert/scoop` 중 하나를 gripper event/action contract로 끌어올리면 논문 스토리가 생긴다.
- 이후 FL은 "same-env solved policy를 federated setting으로 확장"하는 형태로 들어가야 한다.
