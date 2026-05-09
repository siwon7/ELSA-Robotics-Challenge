# Ductor Context Bridge - ELSA/RALPH 2026-05-09

작성 시각: 2026-05-09 10:05 KST

이 파일은 Telegram ductor에서 새 Codex 세션이 현재 대화 맥락을 이어받기 위한 handoff 문서다. 새 세션은 먼저 이 파일과 아래 repo 문서를 읽고, 실행 중인 tmux와 결과 파일을 확인한 뒤 이어서 작업해야 한다.

## 사용자 목표

- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge`에서 RALPH/FLAME 계열 연구를 계속 진행한다.
- same env 4개 task의 SR을 빠르게 끌어올릴 action representation + vision representation을 찾는 것이 최우선이다.
- 목표 task:
  - `close_box`
  - `scoop_with_spatula`
  - `insert_onto_square_peg`
  - `slide_block_to_target`
- 장기 목표는 action/vision ceiling을 먼저 올리고, 이후 federated learning을 붙여 논문화 가능성을 만든다.
- 사용자는 한국어로 짧고 직접적인 보고를 선호한다.

## 현재 중요한 결론

- replay ceiling에서 joint-position absolute 실행은 유효했다.
- 특히 `jp_abs_hold1`은 insert/scoop에서 SR 1.0을 보였고, scoop live pack replay도 SR 1.0이었다.
- 하지만 처음 학습한 `VolumeDP + jp_abs chunk4/exec2 + gripw4`는 약했다.
- 완료 결과:
  - scoop `scoop_volumedp_jpabs_w2_grid16_gripw4_mgr_e100_s0`: SR 0.0
  - slide `slide_volumedp_jpabs_w2_grid16_gripw4_mgr_e100_s0`: SR 0.05
- 따라서 단순히 action space만 바꾸는 것이 아니라 gripper/head/eval hysteresis, low-dim dropout, proprio fusion, EE auxiliary를 같이 손봐야 한다.

## 현재 실행 중인 핵심 시스템

Repo:

```bash
cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge
```

Action search autopilot:

```bash
tmux attach -t action_search_autopilot_20260509
```

Action search managers:

```bash
tmux attach -t action_search_manager_20260508
tmux attach -t action_search_manager_gpu0_20260508
tmux attach -t action_search_manager_gpu1_20260508
tmux attach -t action_search_manager_gpu2_20260508
```

Other useful sessions:

```bash
tmux attach -t power_watch_20260507_moved
tmux attach -t fast_ceiling_search_wait_20260508
```

Autopilot status:

```bash
sed -n '1,220p' /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/action_search_manager_20260508/AUTOPILOT_STATUS.md
```

Autopilot log:

```bash
tail -n 200 /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/action_search_autopilot_20260509/autopilot.log
```

## 가장 최근 구현/커밋

Latest pushed commit:

```text
960cd1c Add autonomous action search autopilot
```

추가된 중요 파일:

- `_workspace/action_search_autopilot_handoff_20260509.md`
- `scripts/action_search_autopilot_20260509.py`
- `scripts/start_action_search_autopilot_20260509_tmux.sh`
- `experiments/action_search_autopilot_20260509/*.yaml`
- `scripts/action_search_manager_20260508_gpu0_queue.tsv`
- `scripts/action_search_manager_20260508_gpu1_queue.tsv`
- `scripts/action_search_manager_20260508_gpu2_queue.tsv`

현재 autopilot은 10분마다 결과/큐를 보고, 큐가 부족하면 다음 후보를 자동 생성/append한다.

## 현재 새로 걸어둔 후보

1차 후보는 replay-faithful action을 학습 가능하게 만든 버전이다.

- action: `joint_position_direct`, `joint_position_absolute`
- horizon: hold-1, `action_chunk_len=1`, `execute_steps=1`
- vision: `volumedp_full_dinov3_depth`
- LoRA: DINOv3 LoRA rank 8
- 추가 안정화:
  - `low_dim_dropout_prob: 0.0`
  - tight VolumeDP bounds
  - `proprio_visual_fusion_mode: gated_global_film`
  - `ee_aux_loss_weight: 1.0`
  - separate gripper head
  - `gripper_loss_weight: 6.0`
  - `gripper_transition_weight: 8.0`
  - `gripper_eval_mode: hysteresis`

이미 시작된 신규 학습:

- GPU0: `close_jpabs_w1_tight_gfilm_eeaux_gripw8_hyst_auto20260509_s0`
- GPU2: `scoop_jpabs_w1_tight_gfilm_eeaux_gripw8_hyst_auto20260509_s0`

기존 진행 중이던 학습:

- GPU1: `insert_volumedp_jpabs_w2_grid16_gripw4_mgr_e100_s0`
- GPU3/main: `close_volumedp_jpabs_w2_grid16_gripw4_mgr_e100_s0`

GPU1이 끝나면 다음 큐:

- `slide_jpabs_w1_tight_gfilm_eeaux_gripw8_hyst_auto20260509_s0`

GPU0/GPU2 큐에도 추가 후보가 남아 있다.

## 검증된 것

다음 검증을 통과했다.

```bash
/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python -m py_compile scripts/action_search_autopilot_20260509.py
/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python scripts/action_search_autopilot_20260509.py --dry-run --min-outstanding-per-queue 2 --max-add 8
/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python scripts/validate_experiment_config.py --config experiments/action_search_autopilot_20260509/close_jpabs_w1_tight_gfilm_eeaux_gripw8_hyst.yaml --normalize
```

전체 candidate template도 runtime config validation 기준으로 통과했다.

## Ductor 상태

ductor는 이미 실행 중이다.

```bash
cd /home/cvlab-dgx/siwon/ductor
HOME=/home/cvlab-dgx/siwon .venv/bin/ductor status
```

확인된 상태:

- pid: `3025187`
- provider: `codex`
- model: `gpt-5.4`
- working dir: `/home/cvlab-dgx/siwon`
- ductor home: `/home/cvlab-dgx/siwon/.ductor`
- Telegram transport 활성화

중요 제한:

- Telegram ductor는 이 API 대화 세션 자체에 입력을 주입하지 않는다.
- 대신 별도 Codex CLI 세션/태스크가 같은 서버와 같은 파일시스템을 보고 작업한다.
- 그래서 이 파일을 먼저 읽게 해야 현재 맥락을 실질적으로 이어받는다.

## Telegram에 보낼 추천 첫 메시지

아래 메시지를 Telegram ductor에 그대로 보내면 된다.

```text
/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge 로 가서
_workspace/ductor_context_bridge_20260509_KR.md 와
_workspace/action_search_autopilot_handoff_20260509.md 를 먼저 읽어.

이 대화의 목적은 same env 4개 task(close_box, scoop_with_spatula, insert_onto_square_peg, slide_block_to_target)의 SR을 0.9 이상으로 올릴 action+vision representation을 찾는 것이다.

현재 action_search_autopilot_20260509 및 action_search_manager_* tmux가 돌고 있으니 상태를 확인하고,
새 결과가 나오면 SR을 요약하고, 낮으면 다음 후보를 큐에 넣고, 필요한 경우 독창적인 action/vision 후보를 추가해라.

현재 학습을 함부로 죽이지 말고, GPU/전력 상태를 먼저 확인한 뒤 안정적으로 이어서 진행해라.
답변은 한국어로 짧고 직접적으로 해라.
```

## 운영 규칙

- 사용자 변경사항은 함부로 되돌리지 말 것.
- 학습 프로세스 kill/restart는 반드시 이유가 명확할 때만 할 것.
- GPU 전력 문제가 있었으므로 한 번에 새 학습을 무리하게 늘리지 말 것.
- 큐와 상태는 문서화하고, 의미 있는 코드/큐 변경은 커밋/푸시할 것.
- 결과 확인 시 최소한 `result.json`의 `sr`와 `mean_per_env_sr`를 같이 볼 것.

## 빠른 상태 확인 명령

```bash
tmux list-sessions
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits
pgrep -af 'train_same_env_bcpolicy_probe.py|action_search_autopilot'
find /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/action_search_manager_20260508 -path '*/result.json' -type f | sort
```

