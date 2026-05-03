#!/usr/bin/env bash
# Phase B: replay-ceiling sweep across new RLBench-supported action modes
# (JP-relative + EE Cartesian via IK + EE Cartesian via Planning).
# Invoked as: ceiling_sweep_phase_b_2026_04_30.sh <task> <gpu_id>
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

unset VIRTUAL_ENV
PATH="/home/cvlab-dgx/anaconda3/condabin:/usr/bin:/bin"
source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
conda activate elsa_challenge
source scripts/prepare_live_eval_env.sh
export CUDA_VISIBLE_DEVICES="$GPU"

run_one() {
    local label="$1"; local mode="$2"; local timeout="$3"
    local out="$OUT_ROOT/${label}.json"
    if [ -f "$out" ]; then
        echo "[$TASK][$label] already done, skipping"
        return 0
    fi
    echo "==================================================================="
    echo "[$TASK][$label] (mode=$mode) starting on GPU $GPU at $(date '+%F %T')"
    echo "==================================================================="
    if python scripts/eval_saved_replay_unified.py \
            --task "$TASK" --pack-dir "$PACK_DIR" --arm-mode "$mode" \
            --max-pack-time-sec "$timeout" --output "$out"; then
        echo "[$TASK][$label] DONE at $(date '+%F %T')"
    else
        echo "[$TASK][$label] FAILED at $(date '+%F %T')" >&2
    fi
}

# Phase B modes: ordered fastest-first so even slow tasks make progress
run_one "jp_rel"               "jp_rel"               60
run_one "ee_ik_abs_world"      "ee_ik_abs_world"      60
run_one "ee_ik_rel_world"      "ee_ik_rel_world"      60
run_one "ee_ik_abs_ee"         "ee_ik_abs_ee"         60
run_one "ee_ik_rel_ee"         "ee_ik_rel_ee"         60
run_one "ee_ik_abs_world_coll" "ee_ik_abs_world_coll" 90
run_one "ee_plan_abs_world"    "ee_plan_abs_world"    180
run_one "ee_plan_rel_world"    "ee_plan_rel_world"    180

echo "[$TASK] Phase B complete at $(date '+%F %T')"
