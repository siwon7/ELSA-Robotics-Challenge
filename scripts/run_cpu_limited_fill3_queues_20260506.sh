#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
LOG_ROOT="${ELSA_FILL3_LOG_ROOT:-$ARTIFACT_ROOT/logs/cpu_limited_fill3_20260506}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
POLL_SEC="${POLL_SEC:-30}"
LOCK_DIR="$LOG_ROOT/.scheduler.lock"

mkdir -p "$LOG_ROOT"

export ELSA_CPU_LIMITS_ENABLED="${ELSA_CPU_LIMITS_ENABLED:-1}"
export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
export ELSA_DATALOADER_WORKERS="${ELSA_DATALOADER_WORKERS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-$ELSA_DATALOADER_WORKERS}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export POLL_SEC

worker_pattern='scripts/(start_action_ablation_queue_tmux|start_jpabs_seedsweep_queue_tmux|start_overnight_queue_pending_tmux|start_recommended_followup_queue_tmux)\.sh --worker'

specs=(
  "action_gpu0|scripts/start_action_ablation_queue_tmux.sh|0"
  "action_gpu1|scripts/start_action_ablation_queue_tmux.sh|1"
  "action_gpu2|scripts/start_action_ablation_queue_tmux.sh|2"
  "action_gpu3|scripts/start_action_ablation_queue_tmux.sh|3"
  "jpabs_gpu0|scripts/start_jpabs_seedsweep_queue_tmux.sh|0"
  "jpabs_gpu1|scripts/start_jpabs_seedsweep_queue_tmux.sh|1"
  "jpabs_gpu2|scripts/start_jpabs_seedsweep_queue_tmux.sh|2"
  "jpabs_gpu3|scripts/start_jpabs_seedsweep_queue_tmux.sh|3"
  "overnight_gpu0|scripts/start_overnight_queue_pending_tmux.sh|0"
  "overnight_gpu1|scripts/start_overnight_queue_pending_tmux.sh|1"
  "overnight_gpu2|scripts/start_overnight_queue_pending_tmux.sh|2"
  "overnight_gpu3|scripts/start_overnight_queue_pending_tmux.sh|3"
  "recommended_gpu0|scripts/start_recommended_followup_queue_tmux.sh|0"
  "recommended_gpu1|scripts/start_recommended_followup_queue_tmux.sh|1"
  "recommended_gpu2|scripts/start_recommended_followup_queue_tmux.sh|2"
  "recommended_gpu3|scripts/start_recommended_followup_queue_tmux.sh|3"
)

acquire_scheduler_lock() {
  local boot_id
  boot_id="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"

  while true; do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      echo "$$" > "$LOCK_DIR/pid"
      echo "$boot_id" > "$LOCK_DIR/boot_id"
      trap 'rm -rf "$LOCK_DIR"' EXIT
      return 0
    fi

    local old_pid old_boot
    old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    old_boot="$(cat "$LOCK_DIR/boot_id" 2>/dev/null || true)"
    if [ "$old_boot" != "$boot_id" ] || [ -z "$old_pid" ] || ! kill -0 "$old_pid" 2>/dev/null; then
      echo "[lock] removing stale scheduler lock old_pid=${old_pid:-unknown} old_boot=${old_boot:-unknown}"
      rm -rf "$LOCK_DIR"
      continue
    fi

    echo "[lock] another fill3 scheduler is already running pid=$old_pid"
    exit 0
  done
}

root_worker_lines() {
  ps -eo pid=,ppid=,args= | awk '
    /scripts\/(start_action_ablation_queue_tmux|start_jpabs_seedsweep_queue_tmux|start_overnight_queue_pending_tmux|start_recommended_followup_queue_tmux)\.sh --worker [0-9]/ {
      pid = $1
      ppid = $2
      match($0, /--worker [0-9]/)
      gpu = substr($0, RSTART + 9, 1)
      rows[pid] = ppid "|" gpu "|" $0
      matched[pid] = 1
    }
    END {
      for (pid in rows) {
        split(rows[pid], parts, "|")
        ppid = parts[1]
        gpu = parts[2]
        if (!(ppid in matched)) {
          print pid, gpu, parts[3]
        }
      }
    }
  '
}

active_worker_count() {
  root_worker_lines | wc -l
}

gpu_busy() {
  local gpu="$1"
  root_worker_lines | awk -v gpu="$gpu" '$2 == gpu { found = 1 } END { exit(found ? 0 : 1) }'
}

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

launch_spec() {
  local label="$1"
  local script="$2"
  local gpu="$3"
  local log="$LOG_ROOT/${label}.log"

  (
    cd "$REPO_ROOT" || exit 1
    echo "=== WORKER START $label script=$script gpu=$gpu $(date '+%F %T') ==="
    bash "$REPO_ROOT/$script" --worker "$gpu"
    status="$?"
    echo "=== WORKER END $label status=$status $(date '+%F %T') ==="
    exit "$status"
  ) >> "$log" 2>&1 &

  echo "launched $label gpu=$gpu pid=$! log=$log"
}

main() {
  cd "$REPO_ROOT" || exit 1
  exec > >(tee -a "$LOG_ROOT/fill3_master.log") 2>&1
  acquire_scheduler_lock

  echo "=== CPU-LIMITED FILL3 QUEUES START $(date '+%F %T') ==="
  echo "limits: max_parallel=$MAX_PARALLEL cores_per_gpu=$ELSA_CPU_CORES_PER_GPU torch_threads=$ELSA_CPU_THREADS_PER_JOB dataloader_workers=$ELSA_DATALOADER_WORKERS batch_size=$BATCH_SIZE"
  log_snapshot start

  local total="${#specs[@]}"
  local launched=()
  local idx
  for idx in "${!specs[@]}"; do
    launched[$idx]=0
  done

  while true; do
    local done_count=0
    for idx in "${!specs[@]}"; do
      if [ "${launched[$idx]}" -eq 1 ]; then
        done_count=$((done_count + 1))
      fi
    done
    if [ "$done_count" -eq "$total" ] && [ "$(active_worker_count)" -eq 0 ]; then
      break
    fi

    local launched_any=0
    while [ "$(active_worker_count)" -lt "$MAX_PARALLEL" ]; do
      local found=0
      local spec label script gpu
      for idx in "${!specs[@]}"; do
        if [ "${launched[$idx]}" -eq 1 ]; then
          continue
        fi

        spec="${specs[$idx]}"
        IFS='|' read -r label script gpu <<< "$spec"
        if gpu_busy "$gpu"; then
          continue
        fi

        launch_spec "$label" "$script" "$gpu"
        launched[$idx]=1
        launched_any=1
        found=1
        sleep 2
        break
      done

      if [ "$found" -eq 0 ]; then
        break
      fi
    done

    echo "[$(date '+%F %T')] active_workers=$(active_worker_count) launched=$done_count/$total"
    if [ "$launched_any" -eq 0 ]; then
      sleep "$POLL_SEC"
    fi
  done

  log_snapshot done
  echo "=== CPU-LIMITED FILL3 QUEUES DONE $(date '+%F %T') ==="
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
