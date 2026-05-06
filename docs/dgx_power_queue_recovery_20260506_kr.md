# DGX 전원 장애 및 학습 큐 복구 기록 (2026-05-06)

## 결론

2026-05-05 밤부터 2026-05-06 오후까지 반복된 중단은 Python 예외나 OOM보다는 전원/PSU/UPS/PDU/보드 계층의 hard reset 가능성이 높다. OS 로그에서는 BMC/IPMI/UPS 이벤트를 직접 볼 수 없어서 확정은 못 하지만, 정상 shutdown 로그 없이 여러 boot가 끊겼고 filesystem dirty bit/ext4 recovery가 반복 확인됐다.

현재 학습 큐는 의도적으로 재시작하지 않는다. 전력 안정화 확인 전에는 추가 학습을 걸지 않는다.

## 확인한 증거

- `journalctl --list-boots` 기준으로 2026-05-06에 `03:17`, `13:30`, `15:17`, `15:36`, `15:53`, `16:56` 근처에서 정상 종료 없이 boot 경계가 끊겼다.
- `last -x`에는 tmux 세션들이 `crash`로 종료된 기록이 남았다.
- 현재 boot에서 `/dev/sda1: Dirty bit is set`, `Fs was not properly unmounted`, `EXT4-fs (md0): recovery complete`가 확인됐다.
- `OOM`, `kernel panic`, `NVRM/Xid`, `thermal`, `MCE`, `pstore` crash artifact는 확인되지 않았다.
- `/dev/ipmi*`, `ipmitool`, UPS/NUT/APC 서비스가 없어 OS 일반 계정에서는 PSU/UPS/BMC 이벤트 로그를 직접 확인할 수 없다.
- GPU power limit은 기본 300W이고, root 권한 없이는 `nvidia-smi -pl` 적용이 불가했다.

## 끊긴 시점과 큐 상태

| 시각 (KST) | 관찰 | 해석 |
| --- | --- | --- |
| 2026-05-05 23:43 | action ablation 4 GPU 작업 중 `close/insert/scoop` 로그가 epoch 중간에서 끊김 | 4 GPU 학습 부하 중 hard reset 가능성 높음 |
| 2026-05-06 03:17 | `03:15` action ablation 4 GPU 재시작 직후 끊김 | 전력 부하 전환/스파이크 가능성 |
| 2026-05-06 13:30 | CPU-limited staged 큐 실행 중 `action_01_gpu1.log`가 epoch 33/50 근처에서 끊김 | 2 GPU 제한 상태에서도 hard reset |
| 2026-05-06 15:17 | fill3 큐가 `active_workers=3`로 지속 실행 중 끊김 | 3 GPU 제한도 불안정 |
| 2026-05-06 15:36 | 학습 로그 갱신 없음, ductor/codex 로그만 확인 | 학습 부하 없이도 불안정 가능성 |
| 2026-05-06 15:53 | 학습 로그 갱신 없음, 이후 16:56 boot | 전원/보드/UPS 쪽 가능성 강화 |

## 남은 실험/큐 상태

전원 이슈 직전 큐는 다음 계열을 돌리려 했다.

- `action_ablation_20260504`: action space ablation, 특히 `jprel`, `jvservo`, `direct`, `jpabs` 계열.
- `jpabs_seedsweep_20260504`: joint-position absolute 계열 seed sweep.
- `overnight_queue`: 기존 VolumeDP/BCPolicy follow-up.
- `recommended_followups_20260504`: 추천 follow-up 실험.
- `power_safe_20260506`: 단일 GPU, batch 8, dataloader worker 0으로 낮춘 복구용 wrapper. 단, GPU 전력 hard cap 없이도 GPU0이 약 282W까지 올라가는 것을 확인해서 중단했다.

실험 결과는 `result.json` 기준으로 2026-05-05 23:13 이후 새 완료 결과가 거의 없고, 2026-05-06의 주요 action 재시도들은 중간에 끊긴 로그가 대부분이다.

## 정상화 원칙

1. 학습 재개 전에 관리자에게 PSU/UPS/PDU/BMC 로그를 요청한다.
2. 가능하면 모든 GPU에 power cap을 먼저 적용한다. 권장 시작점은 150W, 안정하면 180W까지 올린다.
3. power cap 없이 학습을 재개하지 않는다. 단일 GPU에서도 280W대 draw가 관측됐다.
4. 복구는 `MAX_PARALLEL=1`에서 시작하고, 최소 수 시간 안정 후 2, 3으로 올린다.
5. 큐 재시작 시 stale lock과 기존 tmux/session을 확인한다.
6. broad `pkill CoppeliaSim`류 정리는 병렬 worker를 죽일 수 있으므로 금지한다.

## 재개 체크리스트

학습 재개 전:

```bash
tmux ls
pgrep -af 'train_same_env|start_.*queue|fill3|CoppeliaSim'
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits
bash scripts/check_power_health_20260506.sh
```

관리자 권한으로 GPU power cap 적용:

```bash
sudo bash scripts/apply_gpu_power_limit.sh 150
nvidia-smi -q -d POWER | grep -E 'GPU [0-9]|Power Limit|Power Draw'
```

power cap 적용 후 1 GPU 복구 큐 시작:

```bash
tmux new-session -d -s ralph_power_safe_20260506 \
  'cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge && bash scripts/start_power_safe_queue_20260506.sh'
```

상태 확인:

```bash
tmux attach -t ralph_power_safe_20260506
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/power_safe_20260506/fill3_master.log
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/power_health_20260506/power_health.log
```

## 현재 켜둔 모니터

저부하 전원/온도 모니터는 학습 없이 켜둔 상태다.

- tmux session: `power_watch_20260506`
- script: `scripts/monitor_power_health_20260506.sh`
- log: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/power_health_20260506/power_health.log`

다음 reboot 이후에는 아래 명령으로 마지막 모니터 샘플과 현재 boot id를 비교한다.

```bash
bash scripts/check_power_health_20260506.sh
```

## 별도 코드/환경 이슈

`action_01_gpu0.log`에서 live evaluation 중 다음 오류가 확인됐다.

```text
ImportError: cannot import name 'TASKS_PY_FOLDER' from 'colosseum'
```

이 오류는 평가 실패 원인이지만 시스템 hard reset 원인은 아니다. 전원 문제가 안정된 뒤 `colosseum` API/버전 차이를 별도로 고쳐야 한다.

## 권장 다음 단계

- 즉시: 학습 큐 재개 금지, 모니터만 유지.
- 관리자 확인: BMC/IPMI SEL, PSU fault LED, UPS overload/load history, PDU outlet 이벤트를 위 표의 시각 기준으로 확인.
- power cap 적용 후: `start_power_safe_queue_20260506.sh`로 1 GPU만 복구.
- 안정화 후: `MAX_PARALLEL=2`, 이후 `MAX_PARALLEL=3`으로만 점진 확대.
- 논문/실험 측면: 전원 안정 전에는 새 ablation보다 문서화/결과 집계/코드 정리가 우선이다.
