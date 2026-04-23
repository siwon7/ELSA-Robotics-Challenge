#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cv7/haeun/new/ELSA-Robotics-Challenge-siwon-main"

RUN_TAG="${1:?usage: $0 <run_tag> [chunk_sizes_csv]}"
CHUNK_SIZES_CSV="${2:-20,10,5}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
TRAIN_ENV_START="${TRAIN_ENV_START:-0}"
TRAIN_ENV_END="${TRAIN_ENV_END:-399}"
CHUNK_PASSES="${CHUNK_PASSES:-1}"
EPOCHS_PER_CHUNK="${EPOCHS_PER_CHUNK:-1}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"

RUN_LOG="$ROOT/logs/progressive_pipeline/${RUN_TAG}.log"
GUARD_LOG="$ROOT/logs/progressive_pipeline/${RUN_TAG}_guard.log"
STAGE_SUMMARY="$ROOT/results/progressive_pipeline/${RUN_TAG}/stage_checkpoints.txt"

mkdir -p "$(dirname "$GUARD_LOG")"

IFS=',' read -r -a CHUNK_SIZES <<< "$CHUNK_SIZES_CSV"
if [ "${#CHUNK_SIZES[@]}" -eq 0 ]; then
  echo "[guard-error] empty chunk sizes" | tee -a "$GUARD_LOG"
  exit 1
fi

chunk_idx=0
last_train_pid=""
last_seen_ts="$(date '+%F %T')"

echo "[guard-start] run_tag=$RUN_TAG chunk_sizes=$CHUNK_SIZES_CSV passes=$CHUNK_PASSES epochs_per_chunk=$EPOCHS_PER_CHUNK interval=${CHECK_INTERVAL_SEC}s" | tee -a "$GUARD_LOG"

runner_pattern="run_phase1_central400_unseen_chunked_lowmem.sh ${RUN_TAG}"
# Support both old tags (..._stageNNN_...) and new pass tags (..._passXX_stageNNN_...)
train_pattern="train_multi_env_bcpolicy_probe.py.*${RUN_TAG}.*_stage"

read_resume_state() {
  if [ -f "$STAGE_SUMMARY" ] && [ -s "$STAGE_SUMMARY" ]; then
    local pass_token=""
    local last_pass=0
    read -r last_idx _ last_end last_ckpt last_cfg pass_token < <(tail -n 1 "$STAGE_SUMMARY")
    if [[ "$pass_token" =~ ^pass=([0-9]+)$ ]]; then
      last_pass="${BASH_REMATCH[1]}"
    fi
    next_start=$((last_end + 1))
    next_stage=$((last_idx + 1))
    init_ckpt="$last_ckpt"
    init_cfg="$last_cfg"
    next_pass="$last_pass"
    if [ "$next_start" -gt "$TRAIN_ENV_END" ]; then
      next_start="$TRAIN_ENV_START"
      next_pass=$((last_pass + 1))
    fi
  else
    next_start="$TRAIN_ENV_START"
    next_pass=0
    next_stage=0
    init_ckpt=""
    init_cfg=""
  fi
}

run_is_done() {
  rg -q "^\[done\] unseen_eval_json=" "$RUN_LOG" 2>/dev/null
}

had_recent_oom_for_pid() {
  local pid="$1"
  local now_ts
  now_ts="$(date '+%F %T')"
  if [ -n "$pid" ]; then
    if journalctl -k --since "$last_seen_ts" --until "$now_ts" \
      | rg -q "task=python,pid=${pid}|Killed process ${pid} \\(python\\)"; then
      return 0
    fi
  fi
  return 1
}

launch_runner() {
  local chunk_size="$1"
  local start_env="$2"
  local stage_offset="$3"
  local pass_offset="$4"
  local ckpt="$5"
  local cfg="$6"

  tmux kill-session -t phase1-chunked 2>/dev/null || true
  tmux new-session -d -s phase1-chunked \
    "cd $ROOT && \
     TRAIN_GPU=$TRAIN_GPU EVAL_GPU=$EVAL_GPU \
     CHUNK_SIZE=$chunk_size EPOCHS_PER_CHUNK=$EPOCHS_PER_CHUNK CHUNK_PASSES=$CHUNK_PASSES \
     TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE TRAIN_NUM_WORKERS=$TRAIN_NUM_WORKERS TRAIN_NUM_THREADS=$TRAIN_NUM_THREADS \
     TRAIN_ENV_START=$TRAIN_ENV_START TRAIN_ENV_END=$TRAIN_ENV_END \
     START_ENV_ID=$start_env STAGE_INDEX_OFFSET=$stage_offset PASS_INDEX_OFFSET=$pass_offset \
     INIT_CHECKPOINT='$ckpt' FINAL_RESOLVED_CFG='$cfg' \
     EVAL_EPISODES=$EVAL_EPISODES \
     ./scripts/run_phase1_central400_unseen_chunked_lowmem.sh '$RUN_TAG'"

  echo "[guard-restart] chunk_size=$chunk_size pass_offset=$pass_offset start_env=$start_env stage_offset=$stage_offset init_ckpt=${ckpt:-none}" | tee -a "$GUARD_LOG"
}

while true; do
  now_ts="$(date '+%F %T %Z')"
  runner_pid="$(pgrep -f "$runner_pattern" | head -n1 || true)"
  train_pid="$(pgrep -f "$train_pattern" | head -n1 || true)"
  if [ -n "$train_pid" ]; then
    last_train_pid="$train_pid"
  fi

  if [ -n "$runner_pid" ]; then
    echo "[$now_ts] alive runner_pid=$runner_pid train_pid=${train_pid:-none} chunk=${CHUNK_SIZES[$chunk_idx]}" >> "$GUARD_LOG"
    last_seen_ts="$(date '+%F %T')"
    sleep "$CHECK_INTERVAL_SEC"
    continue
  fi

  if run_is_done; then
    echo "[$now_ts] completed: run log has done marker" | tee -a "$GUARD_LOG"
    exit 0
  fi

  if had_recent_oom_for_pid "$last_train_pid"; then
    if [ "$chunk_idx" -lt "$(( ${#CHUNK_SIZES[@]} - 1 ))" ]; then
      chunk_idx=$((chunk_idx + 1))
      read_resume_state
      if [ "$next_pass" -ge "$CHUNK_PASSES" ]; then
        echo "[$now_ts] stages finished, no restart needed (next_pass=$next_pass)" | tee -a "$GUARD_LOG"
        sleep "$CHECK_INTERVAL_SEC"
        continue
      fi
      launch_runner "${CHUNK_SIZES[$chunk_idx]}" "$next_start" "$next_stage" "$next_pass" "$init_ckpt" "$init_cfg"
      last_seen_ts="$(date '+%F %T')"
      sleep "$CHECK_INTERVAL_SEC"
      continue
    fi
    echo "[$now_ts] OOM detected but no smaller chunk left (chunk=${CHUNK_SIZES[$chunk_idx]})" | tee -a "$GUARD_LOG"
    exit 1
  fi

  echo "[$now_ts] runner missing without OOM marker; stop guard for manual check" | tee -a "$GUARD_LOG"
  exit 1
done
