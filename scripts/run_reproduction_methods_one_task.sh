#!/usr/bin/env bash
set -uo pipefail

TASK="${1:?task required}"
GPU_ID="${2:?gpu id required}"
NUM_DEMOS="${3:-3}"
ENV_IDS="${4:-0,1}"
RUN_TAG="${5:-repro_methods_current_repo}"

REPO_ROOT="/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge"
RESULT_ROOT="${REPO_ROOT}/results/${RUN_TAG}/${TASK}"
LOG_ROOT="${REPO_ROOT}/logs/${RUN_TAG}/${TASK}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

cd "${REPO_ROOT}" || exit 1
source scripts/prepare_live_eval_env.sh

run_step() {
  local step_name="$1"
  local output_path="$2"
  shift 2

  if [[ -f "${output_path}" ]]; then
    echo "[skip] ${TASK} ${step_name} -> ${output_path}"
    return 0
  fi

  local log_path="${LOG_ROOT}/${step_name}.log"
  echo "[run] ${TASK} ${step_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n elsa_challenge "$@" |& tee "${log_path}"
}

run_step \
  "live_expert_success" \
  "${RESULT_ROOT}/live_expert_success.json" \
  python scripts/eval_live_expert_reproduction.py \
    --task "${TASK}" \
    --split training \
    --env-ids "${ENV_IDS}" \
    --num-demos "${NUM_DEMOS}" \
    --method expert_success \
    --output "${RESULT_ROOT}/live_expert_success.json"

run_step \
  "saved_pack_replay_stored_joint_vel" \
  "${RESULT_ROOT}/saved_pack_replay_stored_joint_vel.json" \
  python scripts/eval_saved_replay_pack.py \
    --task "${TASK}" \
    --split training \
    --pack-dir "${RESULT_ROOT}" \
    --method stored_joint_vel \
    --output "${RESULT_ROOT}/saved_pack_replay_stored_joint_vel.json"

run_step \
  "saved_pack_replay_finite_diff" \
  "${RESULT_ROOT}/saved_pack_replay_finite_diff.json" \
  python scripts/eval_saved_replay_pack.py \
    --task "${TASK}" \
    --split training \
    --pack-dir "${RESULT_ROOT}" \
    --method finite_diff \
    --output "${RESULT_ROOT}/saved_pack_replay_finite_diff.json"

run_step \
  "saved_pack_replay_target_joint_vel" \
  "${RESULT_ROOT}/saved_pack_replay_target_joint_vel.json" \
  python scripts/eval_saved_replay_pack.py \
    --task "${TASK}" \
    --split training \
    --pack-dir "${RESULT_ROOT}" \
    --method target_joint_vel \
    --output "${RESULT_ROOT}/saved_pack_replay_target_joint_vel.json"

echo "[task_complete] ${TASK}"
