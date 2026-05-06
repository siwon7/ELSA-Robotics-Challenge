#!/usr/bin/env bash
set -euo pipefail

ROOT="${ELSA_POWER_MONITOR_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/power_health_20260506}"
LOG="$ROOT/power_health.log"

if [ ! -f "$LOG" ]; then
  echo "monitor_log_missing=$LOG"
  exit 1
fi

current_boot_id="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
current_boot_time="$(uptime -s 2>/dev/null || true)"
last_header="$(grep '^=== ' "$LOG" | tail -n 1 || true)"
last_boot_id="$(printf '%s\n' "$last_header" | sed -n 's/.*boot_id=\([^ ]*\).*/\1/p')"
last_ts="$(printf '%s\n' "$last_header" | awk '{print $2}')"

echo "current_boot_id=$current_boot_id"
echo "current_boot_time=$current_boot_time"
echo "last_monitor_ts=$last_ts"
echo "last_monitor_boot_id=$last_boot_id"

if [ -n "$last_boot_id" ] && [ "$last_boot_id" != "$current_boot_id" ]; then
  echo "boot_id_changed_since_last_monitor=1"
  echo "interpretation=system rebooted/reset after the last monitor sample"
else
  echo "boot_id_changed_since_last_monitor=0"
  echo "interpretation=no reboot/reset detected since the monitor started on this boot"
fi

echo "active_training_processes:"
pgrep -af 'train_same_env|start_.*queue|fill3|CoppeliaSim' || true

echo "current_gpu_state:"
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "last_monitor_block:"
tail -n 8 "$LOG"
