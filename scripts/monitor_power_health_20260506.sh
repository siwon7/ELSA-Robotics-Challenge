#!/usr/bin/env bash
set -euo pipefail

ROOT="${ELSA_POWER_MONITOR_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/power_health_20260506}"
INTERVAL="${ELSA_POWER_MONITOR_INTERVAL:-5}"
mkdir -p "$ROOT"

LOG="$ROOT/power_health.log"
BOOT_ID_FILE="/proc/sys/kernel/random/boot_id"

echo "power monitor started at $(date -Is)" | tee -a "$LOG"
echo "host=$(hostname) boot_id=$(cat "$BOOT_ID_FILE" 2>/dev/null || true)" | tee -a "$LOG"

while true; do
  ts="$(date -Is)"
  boot_id="$(cat "$BOOT_ID_FILE" 2>/dev/null || true)"
  uptime_s="$(cut -d' ' -f1 /proc/uptime 2>/dev/null || true)"
  loadavg="$(cat /proc/loadavg 2>/dev/null || true)"
  mem="$(free -m | awk 'NR==2 {printf "mem_used_mb=%s mem_total_mb=%s", $3, $2}' 2>/dev/null || true)"

  {
    echo "=== $ts boot_id=$boot_id uptime_s=$uptime_s loadavg=$loadavg $mem"
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1 || true
    for d in /sys/class/hwmon/hwmon*; do
      name="$(cat "$d/name" 2>/dev/null || true)"
      [ "$name" = "coretemp" ] || continue
      printf 'cpu_temps_millic='
      for f in "$d"/temp*_input; do
        [ -e "$f" ] && printf '%s,' "$(cat "$f" 2>/dev/null || true)"
      done
      printf '\n'
    done
  } >> "$LOG"

  sleep "$INTERVAL"
done
