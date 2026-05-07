#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
OUT_ROOT="${TGAC_PREFLIGHT_OUT_ROOT:-$ARTIFACT_ROOT/logs/tgac_20260507/preflight}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
PYTHON_BIN="/home/cvlab-dgx/anaconda3/envs/${ENV_NAME}/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

mkdir -p "$OUT_ROOT"
REPORT="$OUT_ROOT/tgac_preflight_$(date '+%Y%m%d_%H%M%S').log"

exec > >(tee -a "$REPORT") 2>&1

cd "$REPO_ROOT"

echo "=== TGAC PREFLIGHT START $(date '+%F %T') ==="
echo "repo=$REPO_ROOT"
echo "artifact_root=$ARTIFACT_ROOT"
echo "python=$PYTHON_BIN"
echo "git_sha=$(git rev-parse HEAD 2>/dev/null || true)"
echo "branch=$(git branch --show-current 2>/dev/null || true)"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
echo

echo "=== queue status ==="
tmux ls 2>/dev/null || true
tail -n 10 "$ARTIFACT_ROOT/logs/fill4_power_moved_20260507/fill3_master.log" 2>/dev/null || true
echo

echo "=== gpu status ==="
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits || true
echo

echo "=== static checks ==="
python -m compileall \
  elsa_learning_agent \
  federated_elsa_robotics \
  scripts/analyze_gripper_transition_metrics.py \
  scripts/eval_flower_checkpoint_live.py \
  scripts/train_same_env_bcpolicy_probe.py \
  scripts/aggregate_sameenv_sweep_results.py \
  scripts/wave_summary.py
git diff --check
echo

echo "=== config validation ==="
for cfg in \
  experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux_gripw6xk8.yaml \
  experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_jvservo_grid16_eeaux_gripw6xk8.yaml
do
  "$PYTHON_BIN" - "$cfg" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
from elsa_learning_agent.config_validation import validate_runtime_config

cfg_path = Path(sys.argv[1])
cfg = OmegaConf.load(cfg_path)
summary = validate_runtime_config(cfg)
print(
    f"{cfg_path}: gripw={summary['gripper_transition_weight']} "
    f"win={summary['gripper_transition_window']} "
    f"mode={summary['gripper_eval_mode']}"
)
PY
done
echo

echo "=== script help smoke ==="
"$PYTHON_BIN" scripts/analyze_gripper_transition_metrics.py --help >/dev/null
"$PYTHON_BIN" scripts/eval_flower_checkpoint_live.py --help >/dev/null
echo "help checks ok"
echo

echo "=== optional dataset smoke hints ==="
cat <<'EOF'
Dataset shard checks are intentionally not run by default because they can touch many large files.
Run one of these after the active queue is idle:

/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python federated_elsa_robotics/check_dataset_integrity.py \
  --root /mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets/training \
  --task close_box --task insert_onto_square_peg --task scoop_with_spatula --retries 3

/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python federated_elsa_robotics/repro_guard.py \
  --task close_box --split training --env-start 0 --env-stop 1 --retries 3
EOF
echo

echo "=== TGAC PREFLIGHT DONE $(date '+%F %T') ==="
echo "report=$REPORT"
