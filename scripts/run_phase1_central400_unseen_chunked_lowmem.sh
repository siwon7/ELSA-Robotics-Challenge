#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/home/cv7/haeun/new/worktrees/ELSA-Robotics-Challenge-sameenv}"
ART_ROOT="${ART_ROOT:-/home/cv7/haeun/new/ELSA-Robotics-Challenge-siwon-main}"

RUN_TAG="${1:-phase1_multi_env_close_box_central400_unseen_chunked_$(date +%Y%m%d_%H%M%S)}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
TRAIN_ENV_START="${TRAIN_ENV_START:-0}"
TRAIN_ENV_END="${TRAIN_ENV_END:-399}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
EPOCHS_PER_CHUNK="${EPOCHS_PER_CHUNK:-1}"
CHUNK_PASSES="${CHUNK_PASSES:-1}"
START_ENV_ID="${START_ENV_ID:-$TRAIN_ENV_START}"
STAGE_INDEX_OFFSET="${STAGE_INDEX_OFFSET:-0}"
PASS_INDEX_OFFSET="${PASS_INDEX_OFFSET:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
FINAL_RESOLVED_CFG="${FINAL_RESOLVED_CFG:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
UNSEEN_ENV_IDS="${UNSEEN_ENV_IDS:-400,401,402,403,404,405,406,407,408,409}"

LOG_PATH="$ART_ROOT/logs/progressive_pipeline/${RUN_TAG}.log"
UNSEEN_RESULT_DIR="$ART_ROOT/results/progressive_pipeline/$RUN_TAG"
UNSEEN_EVAL_JSON="$UNSEEN_RESULT_DIR/phase1_central400_unseen_eval.json"
STAGE_SUMMARY_TXT="$UNSEEN_RESULT_DIR/stage_checkpoints.txt"

mkdir -p "$(dirname "$LOG_PATH")" "$UNSEEN_RESULT_DIR"

if [ "$STAGE_INDEX_OFFSET" = "0" ] \
  && [ "$PASS_INDEX_OFFSET" = "0" ] \
  && [ "$START_ENV_ID" = "$TRAIN_ENV_START" ] \
  && [ -z "$INIT_CHECKPOINT" ]; then
  : > "$STAGE_SUMMARY_TXT"
else
  touch "$STAGE_SUMMARY_TXT"
fi

export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"

source "$WORKTREE/scripts/prepare_live_eval_env.sh"

echo "[start] run_tag=$RUN_TAG" >> "$LOG_PATH"
echo "[config] chunk_size=$CHUNK_SIZE epochs_per_chunk=$EPOCHS_PER_CHUNK chunk_passes=$CHUNK_PASSES batch_size=$TRAIN_BATCH_SIZE num_workers=$TRAIN_NUM_WORKERS threads=$TRAIN_NUM_THREADS start_env=$START_ENV_ID stage_offset=$STAGE_INDEX_OFFSET pass_offset=$PASS_INDEX_OFFSET init_ckpt=${INIT_CHECKPOINT:-none}" >> "$LOG_PATH"

if [ "$CHUNK_PASSES" -lt 1 ]; then
  echo "[error] CHUNK_PASSES must be >= 1: $CHUNK_PASSES" >> "$LOG_PATH"
  exit 1
fi
if [ "$PASS_INDEX_OFFSET" -lt 0 ] || [ "$PASS_INDEX_OFFSET" -ge "$CHUNK_PASSES" ]; then
  echo "[error] PASS_INDEX_OFFSET out of range: $PASS_INDEX_OFFSET (chunk_passes=$CHUNK_PASSES)" >> "$LOG_PATH"
  exit 1
fi

init_ckpt="$INIT_CHECKPOINT"
final_ckpt="$INIT_CHECKPOINT"
final_resolved_cfg="$FINAL_RESOLVED_CFG"
stage_idx="$STAGE_INDEX_OFFSET"

for ((pass=PASS_INDEX_OFFSET; pass<CHUNK_PASSES; pass++)); do
  if [ "$pass" -eq "$PASS_INDEX_OFFSET" ]; then
    pass_start="$START_ENV_ID"
  else
    pass_start="$TRAIN_ENV_START"
  fi
  echo "[pass-start] pass=$pass start_env=$pass_start" >> "$LOG_PATH"

  for ((start=pass_start; start<=TRAIN_ENV_END; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE - 1))
    if [ "$end" -gt "$TRAIN_ENV_END" ]; then
      end="$TRAIN_ENV_END"
    fi
    train_env_ids="$(seq -s, "$start" "$end")"
    stage_tag="${RUN_TAG}_pass$(printf '%02d' "$pass")_stage$(printf '%03d' "$stage_idx")_env${start}-${end}"

    echo "[stage-start] pass=$pass idx=$stage_idx env_range=${start}-${end} init_ckpt=${init_ckpt:-none}" >> "$LOG_PATH"

    TRAIN_ARGS=(
      "$WORKTREE/scripts/train_multi_env_bcpolicy_probe.py"
      --task close_box
      --train-env-ids "$train_env_ids"
      --eval-env-ids "$start"
      --dataset-config-path "$WORKTREE/experiments/close_box_sameenv_action_onestep_cnn_jpdirect.yaml"
      --epochs "$EPOCHS_PER_CHUNK"
      --milestones "$EPOCHS_PER_CHUNK"
      --batch-size "$TRAIN_BATCH_SIZE"
      --num-workers "$TRAIN_NUM_WORKERS"
      --eval-episodes 1
      --device cuda:0
      --run-name "$stage_tag"
      --output-root "$ART_ROOT/results/multi_env_suite"
      --checkpoint-root "$ART_ROOT/model_checkpoints/multi_env_suite"
    )

    if [ -n "$init_ckpt" ]; then
      TRAIN_ARGS+=(--init-checkpoint "$init_ckpt")
    fi

    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
      "$ELSA_PYTHON_BIN" "${TRAIN_ARGS[@]}" \
      >> "$LOG_PATH" 2>&1

    printf -v EPOCH_PAD "%03d" "$EPOCHS_PER_CHUNK"
    final_ckpt="$ART_ROOT/model_checkpoints/multi_env_suite/close_box/$stage_tag/epoch_${EPOCH_PAD}.pth"
    final_resolved_cfg="$ART_ROOT/results/multi_env_suite/close_box/$stage_tag/resolved_config.yaml"

    if [ ! -f "$final_ckpt" ]; then
      echo "[error] missing checkpoint: $final_ckpt" >> "$LOG_PATH"
      exit 1
    fi

    echo "$stage_idx $start $end $final_ckpt $final_resolved_cfg pass=$pass" >> "$STAGE_SUMMARY_TXT"
    echo "[stage-done] pass=$pass idx=$stage_idx ckpt=$final_ckpt" >> "$LOG_PATH"

    init_ckpt="$final_ckpt"
    stage_idx=$((stage_idx + 1))
  done

  echo "[pass-done] pass=$pass last_ckpt=${final_ckpt:-none}" >> "$LOG_PATH"
done

if [ ! -f "$final_ckpt" ]; then
  echo "[error] final checkpoint not found" >> "$LOG_PATH"
  exit 1
fi
if [ -z "$final_resolved_cfg" ] || [ ! -f "$final_resolved_cfg" ]; then
  echo "[error] final resolved config not found: ${final_resolved_cfg:-empty}" >> "$LOG_PATH"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$ELSA_PYTHON_BIN" "$WORKTREE/scripts/eval_flower_checkpoint_live.py" \
    --model-path "$final_ckpt" \
    --task close_box \
    --dataset-config-path "$final_resolved_cfg" \
    --split eval \
    --env-ids "$UNSEEN_ENV_IDS" \
    --episodes "$EVAL_EPISODES" \
    --device cuda:0 \
    --output "$UNSEEN_EVAL_JSON" \
    >> "$LOG_PATH" 2>&1

echo "[done] run_tag=$RUN_TAG" >> "$LOG_PATH"
echo "[done] final_ckpt=$final_ckpt" >> "$LOG_PATH"
echo "[done] unseen_eval_json=$UNSEEN_EVAL_JSON" >> "$LOG_PATH"
