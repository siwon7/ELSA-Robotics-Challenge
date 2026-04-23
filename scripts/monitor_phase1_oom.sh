#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-phase1_chunked_watch}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
OUT_DIR="${OUT_DIR:-/home/cv7/haeun/new/ELSA-Robotics-Challenge-siwon-main/logs/progressive_pipeline}"
OUT_LOG="$OUT_DIR/${RUN_TAG}_oom_monitor.log"
PID_PATTERN="${PID_PATTERN:-train_multi_env_bcpolicy_probe.py|eval_flower_checkpoint_live.py|run_phase1_central400_unseen_chunked_lowmem.sh}"

mkdir -p "$OUT_DIR"

echo "[start] run_tag=$RUN_TAG interval=${INTERVAL_SEC}s pattern=$PID_PATTERN" | tee -a "$OUT_LOG"

last_check="$(date '+%F %T')"

while true; do
  ts="$(date '+%F %T %Z')"
  ts_plain="$(date '+%F %T')"
  echo "[$ts] ===== monitor tick =====" | tee -a "$OUT_LOG"
  uptime | tee -a "$OUT_LOG"
  free -h | tee -a "$OUT_LOG"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$OUT_LOG"

  if pgrep -af "$PID_PATTERN" | tee -a "$OUT_LOG"; then
    :
  else
    echo "[$ts] no matching training process" | tee -a "$OUT_LOG"
  fi

  journalctl -k --since "$last_check" --until "$ts_plain" \
    | rg -i "oom-kill|out of memory|killed process" \
    | tee -a "$OUT_LOG" || true
  echo | tee -a "$OUT_LOG"

  last_check="$ts_plain"
  sleep "$INTERVAL_SEC"
done
