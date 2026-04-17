#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_PATH="${1:?usage: $0 <training_log_path> [tmux_session_name]}"
SESSION_NAME="${2:-elsa-frcr-bc-gripper-followup}"

TRAIN_CMD_PATTERN="run_flwr_simulation.py --app-dir ${ROOT_DIR} --dataset-config-path dataset_config_frcr_bc_gripper.yaml --dataset-task close_box --policy-name frcr_bc_gripper"
CHECKPOINT_GLOB="/mnt/ext_sdb1/elsa_robotics_challenge/model_checkpoints/close_box/fedavg_FRCRBCGripperPolicy_l-ep_50_ts_0.9_fclients_0.05_round_*.pth"
RESULT_ROOT="${ROOT_DIR}/results/eval_checkpoint/frcr_bc_gripper_final_watch"
FINAL_JSON="${RESULT_ROOT}/final_report.json"

mkdir -p "${RESULT_ROOT}"

while pgrep -f "${TRAIN_CMD_PATTERN}" >/dev/null 2>&1; do
  sleep 60
done

LATEST_MODEL=$(ls -1 ${CHECKPOINT_GLOB} 2>/dev/null | sort -V | tail -n 1 || true)
if [ -z "${LATEST_MODEL}" ]; then
  echo "No checkpoint found matching ${CHECKPOINT_GLOB}" > "${FINAL_JSON}"
  exit 1
fi

cd "${ROOT_DIR}"
source "${ROOT_DIR}/.venv/bin/activate"
export FOLLOWUP_SESSION_NAME="${SESSION_NAME}"

python scripts/eval_checkpoint.py \
  --model-path "${LATEST_MODEL}" \
  --task close_box \
  --device cuda:1 \
  --policy-name frcr_bc_gripper \
  --offline-env-start 400 \
  --offline-env-end 401 \
  --output "${RESULT_ROOT}/offline.json"

ELSA_POLICY_NAME=frcr_bc_gripper \
ELSA_SIM_DEVICE=cuda:1 \
ELSA_SIM_SAVE_VIDEOS=1 \
ELSA_SIM_VIDEO_DIR="${RESULT_ROOT}/videos" \
ELSA_SIM_VIDEO_FPS=20 \
ELSA_OFFLINE_ENV_START=400 \
ELSA_OFFLINE_ENV_END=401 \
ELSA_LIVE_ENV_IDS=400 \
ELSA_NUM_EPISODES_LIVE=5 \
bash scripts/run_eval_checkpoint_online.sh \
  "${LATEST_MODEL}" \
  close_box \
  "${RESULT_ROOT}/online.json" \
  eval

CUDA_VISIBLE_DEVICES=1 python scripts/eval_action_distribution.py \
  --model-path "${LATEST_MODEL}" \
  --task close_box \
  --env-id 400 \
  --device cuda:0 \
  --policy-name frcr_bc_gripper \
  --dataset-config-path dataset_config_frcr_bc_gripper.yaml \
  --output "${RESULT_ROOT}/action_distribution_env400.json"

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

root = Path("results/eval_checkpoint/frcr_bc_gripper_final_watch")
offline = json.loads((root / "offline.json").read_text())
online = json.loads((root / "online.json").read_text())
action = json.loads((root / "action_distribution_env400.json").read_text())

mean_reward = float(online.get("sr", online.get("online", {}).get("mean_reward", 0.0)))
gripper_acc = float(action.get("gripper_acc_thresh_0_5", 0.0))
offline_mse = float(offline.get("mse", offline.get("offline", {}).get("mean_loss", 0.0)))

report = {
    "latest_model": online["model_path"],
    "offline_mse": offline_mse,
    "online_sr": mean_reward,
    "gripper_acc_thresh_0_5": gripper_acc,
    "launch_followup": False,
    "followup_session": None,
}

if mean_reward < 0.2 or gripper_acc < 0.9:
    session_name = os.environ.get("FOLLOWUP_SESSION_NAME", "elsa-frcr-bc-gripper-v2")
    cmd = (
        f"cd {Path.cwd()} && source .venv/bin/activate && "
        "ELSA_TASK=close_box "
        "ELSA_STORAGE_ROOT=/mnt/ext_sdb1/elsa_robotics_challenge "
        "ELSA_USE_WANDB=false "
        "bash scripts/run_task_frcr_bc_gripper_v2_eval_3gpu.sh "
        f"2>&1 | tee logs/close_box_frcr_bc_gripper_v2_$(date +%Y%m%d_%H%M%S).log"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd], check=True)
    report["launch_followup"] = True
    report["followup_session"] = session_name

(root / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
