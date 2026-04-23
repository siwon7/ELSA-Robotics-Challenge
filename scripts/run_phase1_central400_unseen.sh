#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/home/cv7/haeun/new/worktrees/ELSA-Robotics-Challenge-sameenv}"
ART_ROOT="${ART_ROOT:-/home/cv7/haeun/new/ELSA-Robotics-Challenge-siwon-main}"

RUN_TAG="${1:-phase1_multi_env_close_box_central400_unseen_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-20}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
TRAIN_ENV_IDS="${TRAIN_ENV_IDS:-$(seq -s, 0 399)}"
UNSEEN_ENV_IDS="${UNSEEN_ENV_IDS:-400,401,402,403,404,405,406,407,408,409}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-}"

LOG_PATH="$ART_ROOT/logs/progressive_pipeline/${RUN_TAG}.log"
TRAIN_RESULT_DIR="$ART_ROOT/results/multi_env_suite/close_box/$RUN_TAG"
TRAIN_CKPT_DIR="$ART_ROOT/model_checkpoints/multi_env_suite/close_box/$RUN_TAG"
UNSEEN_RESULT_DIR="$ART_ROOT/results/progressive_pipeline/$RUN_TAG"
UNSEEN_EVAL_JSON="$UNSEEN_RESULT_DIR/phase1_central400_unseen_eval.json"

mkdir -p "$(dirname "$LOG_PATH")" "$UNSEEN_RESULT_DIR"

# Disable tqdm bars to keep logs compact/stable in long runs.
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

source "$WORKTREE/scripts/prepare_live_eval_env.sh"

if [ -n "$TRAIN_NUM_THREADS" ]; then
  export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
  export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
  export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$ELSA_PYTHON_BIN" "$WORKTREE/scripts/train_multi_env_bcpolicy_probe.py" \
    --task close_box \
    --train-env-ids "$TRAIN_ENV_IDS" \
    --eval-env-ids "0" \
    --dataset-config-path "$WORKTREE/experiments/close_box_sameenv_action_onestep_cnn_jpdirect.yaml" \
    --epochs "$EPOCHS" \
    --milestones "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --eval-episodes 1 \
    --device cuda:0 \
    --run-name "$RUN_TAG" \
    --output-root "$ART_ROOT/results/multi_env_suite" \
    --checkpoint-root "$ART_ROOT/model_checkpoints/multi_env_suite" \
    >> "$LOG_PATH" 2>&1

printf -v EPOCH_PAD "%03d" "$EPOCHS"
FINAL_CKPT="$TRAIN_CKPT_DIR/epoch_${EPOCH_PAD}.pth"
RESOLVED_CFG="$TRAIN_RESULT_DIR/resolved_config.yaml"

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$ELSA_PYTHON_BIN" "$WORKTREE/scripts/eval_flower_checkpoint_live.py" \
    --model-path "$FINAL_CKPT" \
    --task close_box \
    --dataset-config-path "$RESOLVED_CFG" \
    --split eval \
    --env-ids "$UNSEEN_ENV_IDS" \
    --episodes "$EVAL_EPISODES" \
    --device cuda:0 \
    --output "$UNSEEN_EVAL_JSON" \
    >> "$LOG_PATH" 2>&1

echo "[done] run_tag=$RUN_TAG" >> "$LOG_PATH"
echo "[done] unseen_eval_json=$UNSEEN_EVAL_JSON" >> "$LOG_PATH"
