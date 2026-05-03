#!/usr/bin/env bash
# Per-task replay-ceiling sweep across all currently-implemented action modes.
# Invoked as: ceiling_sweep_2026_04_30.sh <task> <gpu_id>
# Reads packs from results/replay_pack_train01x5_v1/<task>/
# Writes per-mode JSON to results/ceiling_sweep_2026_04_30/<task>/<mode>.json
set -euo pipefail

TASK="$1"
GPU="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PACK_DIR="results/replay_pack_train01x5_v1/$TASK"
OUT_ROOT="results/ceiling_sweep_2026_04_30/$TASK"
mkdir -p "$OUT_ROOT"

if [ ! -d "$PACK_DIR" ]; then
    echo "missing pack dir: $PACK_DIR" >&2
    exit 1
fi

# clear any inherited venv (ductor) before activating conda env
unset VIRTUAL_ENV
PATH="/home/cvlab-dgx/anaconda3/condabin:/usr/bin:/bin"
source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
conda activate elsa_challenge
source scripts/prepare_live_eval_env.sh
export CUDA_VISIBLE_DEVICES="$GPU"

run_one() {
    local label="$1"; shift
    local out="$OUT_ROOT/${label}.json"
    if [ -f "$out" ]; then
        echo "[$TASK][$label] already done, skipping"
        return 0
    fi
    echo "==================================================================="
    echo "[$TASK][$label] starting on GPU $GPU at $(date '+%F %T')"
    echo "  cmd: $*"
    echo "==================================================================="
    if "$@" --output "$out"; then
        echo "[$TASK][$label] DONE at $(date '+%F %T')"
    else
        echo "[$TASK][$label] FAILED at $(date '+%F %T')" >&2
    fi
}

# --- JV modes ---
run_one "jv_stored" \
    python scripts/eval_saved_replay_pack.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --method stored_joint_vel

run_one "jv_finite_diff" \
    python scripts/eval_saved_replay_pack.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --method finite_diff

# --- JP modes (absolute_mode=True env) ---
run_one "jp_absolute" \
    python scripts/eval_saved_replay_joint_position.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --mode absolute

run_one "jp_delta_naive" \
    python scripts/eval_saved_replay_joint_position.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --mode delta

run_one "jp_interp2" \
    python scripts/eval_saved_replay_joint_position.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --mode interp2

run_one "jp_interp3" \
    python scripts/eval_saved_replay_joint_position.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --mode interp3

# --- JP -> JV servo modes ---
run_one "jp_servo_g20" \
    python scripts/eval_saved_replay_joint_position.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --mode absolute \
        --benchmark-joint-velocity-servo --servo-gain 20

run_one "jp_servo_g40" \
    python scripts/eval_saved_replay_joint_position.py \
        --task "$TASK" --pack-dir "$PACK_DIR" --mode absolute \
        --benchmark-joint-velocity-servo --servo-gain 40

echo "[$TASK] all modes complete at $(date '+%F %T')"
