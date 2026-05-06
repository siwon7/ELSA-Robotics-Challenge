#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="$ARTIFACT_ROOT/logs/cpu_limited_staged_20260506"
STAGE_COOLDOWN_SEC="${STAGE_COOLDOWN_SEC:-120}"

mkdir -p "$LOG_ROOT"

export ELSA_CPU_LIMITS_ENABLED="${ELSA_CPU_LIMITS_ENABLED:-1}"
export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
export ELSA_DATALOADER_WORKERS="${ELSA_DATALOADER_WORKERS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-$ELSA_DATALOADER_WORKERS}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export POLL_SEC="${POLL_SEC:-120}"

log_snapshot() {
  local label="$1"

  echo "=== SNAPSHOT $label $(date '+%F %T') ==="
  uptime || true
  cat /proc/loadavg || true
  cat /proc/pressure/cpu || true
  nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits || true
  journalctl -b -k --since "10 minutes ago" --no-pager \
    | grep -Ei "mce|machine check|hardware error|thermal|temperature|throttl|critical|watchdog|soft lockup|hard lockup|panic|nmi|overheat|powercap|xid|nvrm" || true
}

run_group() {
  local label="$1"
  local script="$2"
  shift 2
  local gpus=("$@")
  local pids=()
  local gpu

  echo "=== GROUP START $label script=$script gpus=${gpus[*]} $(date '+%F %T') ==="
  log_snapshot "${label}_before"

  for gpu in "${gpus[@]}"; do
    (
      cd "$REPO_ROOT" || exit 1
      echo "=== WORKER START $label gpu=$gpu $(date '+%F %T') ==="
      bash "$script" --worker "$gpu"
      status="$?"
      echo "=== WORKER END $label gpu=$gpu status=$status $(date '+%F %T') ==="
      exit "$status"
    ) >> "$LOG_ROOT/${label}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "launched $label gpu=$gpu pid=${pids[-1]} log=$LOG_ROOT/${label}_gpu${gpu}.log"
  done

  local failed=0
  local idx
  for idx in "${!pids[@]}"; do
    gpu="${gpus[$idx]}"
    if wait "${pids[$idx]}"; then
      echo "worker ok: $label gpu=$gpu"
    else
      local status="$?"
      echo "worker failed: $label gpu=$gpu status=$status"
      failed=1
    fi
  done

  log_snapshot "${label}_after"
  echo "=== GROUP END $label failed=$failed $(date '+%F %T') ==="
  sleep "$STAGE_COOLDOWN_SEC"
  return "$failed"
}

main() {
  cd "$REPO_ROOT" || exit 1
  exec > >(tee -a "$LOG_ROOT/staged_master.log") 2>&1

  echo "=== CPU-LIMITED STAGED QUEUES START $(date '+%F %T') ==="
  echo "limits: cores_per_gpu=$ELSA_CPU_CORES_PER_GPU torch_threads=$ELSA_CPU_THREADS_PER_JOB dataloader_workers=$ELSA_DATALOADER_WORKERS batch_size=$BATCH_SIZE"

  run_group action_01 "$REPO_ROOT/scripts/start_action_ablation_queue_tmux.sh" 0 1 || true
  run_group action_23 "$REPO_ROOT/scripts/start_action_ablation_queue_tmux.sh" 2 3 || true
  run_group jpabs_01 "$REPO_ROOT/scripts/start_jpabs_seedsweep_queue_tmux.sh" 0 1 || true
  run_group jpabs_23 "$REPO_ROOT/scripts/start_jpabs_seedsweep_queue_tmux.sh" 2 3 || true
  run_group overnight_01 "$REPO_ROOT/scripts/start_overnight_queue_pending_tmux.sh" 0 1 || true
  run_group overnight_23 "$REPO_ROOT/scripts/start_overnight_queue_pending_tmux.sh" 2 3 || true
  run_group recommended_01 "$REPO_ROOT/scripts/start_recommended_followup_queue_tmux.sh" 0 1 || true
  run_group recommended_23 "$REPO_ROOT/scripts/start_recommended_followup_queue_tmux.sh" 2 3 || true

  echo "=== CPU-LIMITED STAGED QUEUES DONE $(date '+%F %T') ==="
}

main "$@"
