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

## 2GPU fallback - 2026-05-07

4GPU 재시작 후 다시 hard reset이 발생했다.

- 4GPU 세션: `ralph_fill4_20260507`
- 실행 시작: 2026-05-07 00:17 KST
- 시스템 재부팅: 2026-05-07 00:26 KST
- 마지막 master heartbeat: 2026-05-07 00:23:24 KST
- 마지막 power monitor 구간에서 GPU 온도는 약 52-57C라 thermal 원인은 약하다.
- GPU 전력 샘플은 4GPU 합산 최대 약 1036W까지 관측됐다. 5초 샘플이라 순간 피크는 더 높을 수 있다.

사용자가 장기간 CoRL 작업을 병행해야 하므로, 이후에는 2개 GPU만 동시에 사용하는 conservative queue로 둔다.

새 2GPU 세션:

- tmux session: `ralph_fill2_20260507`
- log root: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill2_20260507`
- master log: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill2_20260507/fill3_master.log`
- scheduler: `scripts/run_cpu_limited_fill3_queues_20260506.sh`
- parallelism: `MAX_PARALLEL=2`
- batch size: `BATCH_SIZE=16`
- dataloader workers: `ELSA_DATALOADER_WORKERS=1`
- CPU threads per job: `ELSA_CPU_THREADS_PER_JOB=1`

재시작 커맨드:

```bash
tmux new-session -d -s ralph_fill2_20260507 \
  "cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge && \
   ELSA_FILL3_LOG_ROOT=/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill2_20260507 \
   MAX_PARALLEL=2 \
   BATCH_SIZE=16 \
   ELSA_CPU_CORES_PER_GPU=4 \
   ELSA_CPU_THREADS_PER_JOB=1 \
   ELSA_DATALOADER_WORKERS=1 \
   NUM_WORKERS=1 \
   POLL_SEC=60 \
   bash scripts/run_cpu_limited_fill3_queues_20260506.sh"
```

상태 확인:

```bash
tmux attach -t ralph_fill2_20260507
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill2_20260507/fill3_master.log
```

## 4GPU retry after power outlet move - 2026-05-07

2GPU fallback도 2026-05-07 03:23 KST 근처에 hard reset으로 종료됐다.
사용자가 전원 위치를 옮긴 뒤 4GPU queue를 다시 테스트한다.

새 4GPU 세션:

- tmux session: `ralph_fill4_power_moved_20260507`
- power monitor tmux session: `power_watch_20260507_moved`
- log root: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507`
- master log: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507/fill3_master.log`
- scheduler: `scripts/run_cpu_limited_fill3_queues_20260506.sh`
- parallelism: `MAX_PARALLEL=4`
- batch size: `BATCH_SIZE=16`
- dataloader workers: `ELSA_DATALOADER_WORKERS=1`
- CPU threads per job: `ELSA_CPU_THREADS_PER_JOB=1`

재시작 커맨드:

```bash
tmux new-session -d -s ralph_fill4_power_moved_20260507 \
  "cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge && \
   ELSA_FILL3_LOG_ROOT=/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/fill4_power_moved_20260507 \
   MAX_PARALLEL=4 \
   BATCH_SIZE=16 \
   ELSA_CPU_CORES_PER_GPU=4 \
   ELSA_CPU_THREADS_PER_JOB=1 \
   ELSA_DATALOADER_WORKERS=1 \
   NUM_WORKERS=1 \
   POLL_SEC=60 \
   bash scripts/run_cpu_limited_fill3_queues_20260506.sh"
```
