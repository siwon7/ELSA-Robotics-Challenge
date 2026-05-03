#!/usr/bin/env bash
# Poll for w2_grid16 done markers; once all 4 tasks have done markers,
# launch the w3_tightbnds wave. Logs to a known file so we can tail it.
set -u

REPO=/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge
LOGDIR=/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/volumedp_full_ablations
W2_TASKS=(slide_sameenv_volumedp_full_w2_grid16 close_box_sameenv_volumedp_full_w2_grid16 insert_sameenv_volumedp_full_w2_grid16 scoop_sameenv_volumedp_full_w2_grid16)

while true; do
  all_done=1
  for t in "${W2_TASKS[@]}"; do
    if [ ! -f "$LOGDIR/${t}.done" ]; then
      all_done=0
      break
    fi
  done
  if [ "$all_done" -eq 1 ]; then
    echo "[$(date '+%F %T')] all W2 done; launching W3 tightbnds"
    cd "$REPO" && bash scripts/run_volumedp_full_wave.sh w3_tightbnds
    exit 0
  fi
  sleep 60
done
