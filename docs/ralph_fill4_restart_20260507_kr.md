# Ralph 4GPU 큐 재시작 기록 - 2026-05-07

## 상황 요약

- 기록 시각: 2026-05-07 00:04 KST
- 기존 4GPU 큐 대상 planned run: 42개
- 완료 결과 보유: 27개
- 기록 시점 실행 중: 1개
- 미완료: 14개
- 현재 실행 중인 1GPU safe run을 내리고 4GPU로 재시작하면, 결과가 없는 run 기준으로 총 15개를 다시 완료해야 한다.

## 현재 내릴 run

- `action_ablation_20260504 / slide_block_to_target / slide_jprel_w2_direct_grid16_eeaux_e50_s0`
- 내리기 직전 상태: `epoch 3/50`
- 결과 파일 없음. 중단 후에는 처음부터 다시 실행되는 것으로 취급한다.

## 완료된 주요 결과

| Queue | Task | Run | SR |
| --- | --- | --- | --- |
| `relative_action_20260504` | `slide_block_to_target` | `slide_jprel_w4_direct_grid16_eeaux_e50_s0` | 0.85 |
| `overnight_queue` | `slide_block_to_target` | `slide_volumedp_w4_eeaux_e50` | 0.60 |
| `action_ablation_20260504` | `slide_block_to_target` | `slide_jprel_w4_jvservo_grid16_eeaux_e50_s0` | 0.55 |
| `overnight_queue` | `slide_block_to_target` | `slide_volumedp_w4_e100` | 0.50 |
| `overnight_queue` | `slide_block_to_target` | `slide_volumedp_w4_film_eeaux_e50` | 0.30 |
| `overnight_queue` | `close_box` | `close_volumedp_w4_proprio_film_e50` | 0.20 |
| `overnight_queue` | `close_box` | `close_volumedp_w4_eeaux_5env_e10` | 0.16 mean |

Notes:
- `paperfaithful_20260504`는 4개 task 모두 완료됐지만 SR은 모두 0.0이다.
- `relative_action_20260504`는 slide만 SR 0.85이고 close/insert/scoop은 0.0이다.
- 현재까지 insert/scoop의 성공 신호는 거의 없다.

## 결과가 없는 남은 run

### Action Ablation

- `slide_block_to_target / slide_jprel_w2_direct_grid16_eeaux_e50_s0`
- `close_box / close_jprel_w4_jvservo_grid16_eeaux_e50_s0`
- `close_box / close_jprel_w2_direct_grid16_eeaux_e50_s0`
- `insert_onto_square_peg / insert_jprel_w4_jvservo_grid16_eeaux_e50_s0`
- `insert_onto_square_peg / insert_jprel_w2_direct_grid16_eeaux_e50_s0`
- `scoop_with_spatula / scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0`
- `scoop_with_spatula / scoop_jprel_w2_direct_grid16_eeaux_e50_s0`

### Overnight Queue

- `insert_onto_square_peg / insert_volumedp_w4_eeaux_e50`

### Recommended Followups

- `slide_block_to_target / slide_volumedp_w4_eeaux_5env_e10_s0`
- `close_box / close_volumedp_w4_eeaux_e100_s0`

Note:
- `close_volumedp_w4_eeaux_e100_s0`는 조건부 run이다. 기준인 `close_volumedp_w4_eeaux_e50`의 SR이 0.05라 스크립트상 skip될 가능성이 높다.

### JPABS / Seed Sweep

- `slide_block_to_target / slide_w2_jvdirect_grid16_e50_s1`
- `slide_block_to_target / slide_jpabs_w2_grid16_e50_s0`
- `slide_block_to_target / slide_w2_jvdirect_grid16_e50_s2`
- `insert_onto_square_peg / insert_jpabs_w2_grid16_e50_s0`
- `scoop_with_spatula / scoop_jpabs_w2_grid16_e50_s0`

## 재시작 설정

새 4GPU 세션:

- tmux session: `ralph_fill4_20260507`
- log root: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_20260507`
- master log: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_20260507/fill3_master.log`
- scheduler: `scripts/run_cpu_limited_fill3_queues_20260506.sh`
- parallelism: `MAX_PARALLEL=4`
- batch size: `BATCH_SIZE=16`
- dataloader workers: `ELSA_DATALOADER_WORKERS=1`
- CPU threads per job: `ELSA_CPU_THREADS_PER_JOB=1`

재시작 커맨드:

```bash
tmux new-session -d -s ralph_fill4_20260507 \
  "cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge && \
   ELSA_FILL3_LOG_ROOT=/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_20260507 \
   MAX_PARALLEL=4 \
   BATCH_SIZE=16 \
   ELSA_CPU_CORES_PER_GPU=4 \
   ELSA_CPU_THREADS_PER_JOB=1 \
   ELSA_DATALOADER_WORKERS=1 \
   NUM_WORKERS=1 \
   POLL_SEC=60 \
   bash scripts/run_cpu_limited_fill3_queues_20260506.sh"
```

복구 방법:

1. 서버가 꺼지면 부팅 후 `tmux ls`와 `ps -eo pid,args | grep train_same_env_bcpolicy_probe`로 생존 프로세스를 확인한다.
2. 위 `master log`에서 마지막으로 launched 된 worker와 active state를 확인한다.
3. 같은 재시작 커맨드를 다시 실행하면 이미 `result.json`이 있는 run은 skip된다.

주의:
- GPU power cap은 적용하지 않았다. root 권한 없이 `nvidia-smi -pl`을 설정할 수 없기 때문이다.
- 4GPU 동시 학습은 전원/PSU/UPS/PDU 문제를 다시 유발할 수 있다.
