# Experiment Queue Record 2026-05-04

기준 저장소: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge`

기준 시각: 2026-05-04 KST

## 현재 활성 큐

### 1. `overnight_pending_16`

목적: 기존 overnight sweep의 마무리. same-env와 5-env 짧은 probe를 섞어서 VolumeDP-full, EE auxiliary, proprio FiLM, JP-servo/chunk 설정의 신호를 확인한다.

완료된 주요 결과:

| Run | Task | SR | 요약 |
| --- | --- | ---: | --- |
| `slide_volumedp_w4_eeaux_e50` | `slide_block_to_target` | 0.60 | 현재 가장 강한 positive signal. |
| `slide_volumedp_w4_e100` | `slide_block_to_target` | 0.50 | VolumeDP-full same-env가 slide에서 작동함. |
| `slide_volumedp_w4_film_eeaux_e50` | `slide_block_to_target` | 0.30 | FiLM+EE aux는 EE aux 단독보다 약함. |
| `close_volumedp_w4_proprio_film_e50` | `close_box` | 0.20 | close에서 약한 positive signal. |
| `close_volumedp_w4_eeaux_e50` | `close_box` | 0.05 | close EE aux 단독은 약함. |
| `insert_baseline_chunk4exec2_e100` | `insert_onto_square_peg` | 0.00 | insert는 raw JP-servo/chunk로 아직 해결 안 됨. |
| `scoop_volumedp_w4_eeaux_e50` | `scoop_with_spatula` | 0.00 | scoop도 현 action interface로는 아직 실패. |

잔여/진행 중 항목:

| Run | Task | 목적 |
| --- | --- | --- |
| `close_volumedp_w4_eeaux_5env_e10` | `close_box` | close EE aux가 5-env에서 완전히 무너지는지 확인. |
| `insert_volumedp_w4_eeaux_e50` | `insert_onto_square_peg` | insert same-env에서 VolumeDP+EE aux가 신호를 내는지 확인. |
| `insert_volumedp_w4_eeaux_5env_e10` | `insert_onto_square_peg` | insert 5-env short probe. |
| `scoop_volumedp_w4_eeaux_5env_e10` | `scoop_with_spatula` | scoop 5-env short probe. |

### 2. `recommended_followups_wait`

스크립트: `scripts/start_recommended_followup_queue_tmux.sh`

목적: `slide_volumedp_w4_eeaux_e50 = 0.60 SR`이 재현 가능한 신호인지 확인한다. 기존 overnight/paperclose blocker가 끝난 뒤 `recommended_followups_20260504` 세션을 자동으로 시작한다.

예정 항목:

| Run | Task | 목적 |
| --- | --- | --- |
| `slide_volumedp_w4_eeaux_e100_s0` | `slide_block_to_target` | e50 winner를 e100으로 확인. |
| `slide_volumedp_w4_eeaux_e50_s1` | `slide_block_to_target` | seed 1 반복. |
| `slide_volumedp_w4_eeaux_e50_s2` | `slide_block_to_target` | seed 2 반복. |
| `slide_volumedp_w4_eeaux_5env_e10_s0` | `slide_block_to_target` | same-env winner의 5-env collapse 여부 확인. |
| `close_volumedp_w4_eeaux_e100_s0` | `close_box` | 조건부 실행. `close_volumedp_w4_eeaux_e50 >= 0.15 SR`이면 실행, 현재 결과 0.05라 skip 예상. |

### 3. `paperfaithful_wait`

스크립트: `scripts/start_paperfaithful_queue_tmux.sh`

목적: 최신 proposal에 가장 충실한 4-task same-env sanity run. `recommended_followups`까지 끝난 뒤 `paperfaithful_20260504` 세션을 자동 시작한다.

공통 설정:

- `VolumeDP-full`
- frozen `DINOv3 + Depth-Anything`
- LoRA rank 8
- temporal RGB pair
- `24^3` voxel grid
- 200 spatial tokens
- proxy goal token
- EE auxiliary BCE
- 100-step diffusion
- separate gripper head
- `JP-servo g=20`
- `action_chunk_len=4`, `receding_horizon_execute_steps=2`

예정 항목:

| Run | Task | 목적 |
| --- | --- | --- |
| `slide_volumedp_jpservo_paperfaithful_v2_e50_s0` | `slide_block_to_target` | 논문식 JP-servo가 slide의 JV winner를 대체할 수 있는지 확인. |
| `close_volumedp_jpservo_paperfaithful_v2_e50_s0` | `close_box` | close에서 paper-faithful action/3D 설정 확인. |
| `insert_volumedp_jpservo_paperfaithful_v2_e50_s0` | `insert_onto_square_peg` | contact-rich insert의 핵심 sanity run. |
| `scoop_volumedp_jpservo_paperfaithful_v2_e50_s0` | `scoop_with_spatula` | contact-rich scoop의 핵심 sanity run. |

## Paperclose V1 주의점

`scripts/wait_and_run_volumedp_paperclose.sh`의 기존 실행은 GPU0가 비자마자 시작됐지만, 환경 Python이 잘못 잡혀 `numpy` import 단계에서 실패했다. 결과 파일은 생성되지 않았다.

보정:

- `scripts/wait_and_run_volumedp_paperclose.sh`
- `scripts/start_recommended_followup_queue_tmux.sh`
- `scripts/start_paperfaithful_queue_tmux.sh`

위 세 스크립트는 이제 `/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python`을 명시적으로 사용한다.

## 현재 해석

가장 강한 신호는 `slide + VolumeDP-full + EE aux`이다. `insert`와 `scoop`은 RMSE가 줄어도 SR이 0에 머무는 패턴이 반복되고 있어, raw joint-position/JP-servo regression을 더 오래 학습시키는 것보다 waypoint/phase-action 계열로 넘어가는 것이 다음 우선순위다.
