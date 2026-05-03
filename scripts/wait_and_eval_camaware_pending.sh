#!/usr/bin/env bash
# Watch the in-progress camera-aware training tmux windows; once each one
# reaches the post-train eval phase (which will fail with a dim mismatch
# because the in-memory process_obs is the pre-patch version), the
# checkpoint is already saved. We then re-run the eval with the fixed
# inference path (ELSA_INCLUDE_CAMERA_IN_STATE=1).
#
# Tasks watched: close_box, insert_onto_square_peg, scoop_with_spatula
# (slide was relaunched separately and will eval cleanly itself.)
set -u

REPO=/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge
SCRIPT_DIR="$REPO/scripts"
LOGDIR="$REPO/logs"
CKPT_ROOT="$REPO/model_checkpoints/same_env_bcpolicy_probes"

declare -A CFG=(
  [close_box]="experiments/close_box_camaware_dino_depth_diffusion_lora8.yaml"
  [insert_onto_square_peg]="experiments/insert_camaware_dino_depth_diffusion_lora8.yaml"
  [scoop_with_spatula]="experiments/scoop_camaware_dino_depth_diffusion_lora8.yaml"
  [slide_block_to_target]="experiments/slide_camaware_dino_depth_diffusion_lora8.yaml"
)
declare -A GPU=(
  [close_box]=1
  [insert_onto_square_peg]=2
  [scoop_with_spatula]=3
  [slide_block_to_target]=0
)
declare -A DONE
DONE[close_box]=0
DONE[insert_onto_square_peg]=0
DONE[scoop_with_spatula]=0
DONE[slide_block_to_target]=0

ready_for_eval() {
  local task="$1"
  local log="$LOGDIR/${task}_camaware_v1.log"
  [ -f "$log" ] || return 1
  local ckpt="$CKPT_ROOT/$task/${task}_camaware_v1/env_000.pth"
  [ -f "$ckpt" ] || return 1
  # train script saves the checkpoint right before online_evaluation, then
  # the in-memory pre-patch process_obs causes an exception. Either signal
  # is sufficient: a Traceback in the log, OR the train loop reached 50/50.
  if grep -qE "epoch=50/50|RuntimeError|Traceback|FAILED" "$log"; then
    return 0
  fi
  return 1
}

run_eval() {
  local task="$1"
  local cfg="${CFG[$task]}"
  local gpu="${GPU[$task]}"
  local run_name="${task}_camaware_v1"
  local logfile="$LOGDIR/${run_name}_eval_rerun.log"
  echo "=== $(date '+%F %T') eval rerun start: $task on GPU $gpu ===" | tee "$logfile"
  (
    unset VIRTUAL_ENV
    PATH="/home/cvlab-dgx/anaconda3/condabin:/usr/bin:/bin"
    source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
    conda activate elsa_challenge
    source "$SCRIPT_DIR/prepare_live_eval_env.sh"
    export ELSA_INCLUDE_CAMERA_IN_STATE=1
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$REPO"
    python scripts/eval_camaware_checkpoint.py \
      --task "$task" \
      --dataset-config-path "$cfg" \
      --run-name "$run_name" \
      --eval-episodes 20 \
      --device cuda:0 \
      --seed 0
  ) >> "$logfile" 2>&1
  local rc=$?
  echo "=== $(date '+%F %T') eval rerun end ($rc): $task ===" | tee -a "$logfile"
  return $rc
}

while true; do
  all_done=1
  for task in close_box insert_onto_square_peg scoop_with_spatula slide_block_to_target; do
    if [ "${DONE[$task]}" -eq 1 ]; then
      continue
    fi
    all_done=0
    if ready_for_eval "$task"; then
      echo "[$(date '+%F %T')] $task is ready for eval rerun"
      if run_eval "$task"; then
        DONE[$task]=1
        echo "[$(date '+%F %T')] $task eval rerun OK"
      else
        echo "[$(date '+%F %T')] $task eval rerun FAILED, will retry next loop"
      fi
    fi
  done
  if [ "$all_done" -eq 1 ]; then
    echo "[$(date '+%F %T')] all camera-aware eval reruns complete"
    exit 0
  fi
  sleep 60
done
