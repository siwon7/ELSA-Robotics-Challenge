#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ELSA_ARTIFACT_ROOT:-/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts}"
RESULT_ROOT="$ARTIFACT_ROOT/results/action_search_manager_20260508"
CKPT_ROOT="$ARTIFACT_ROOT/model_checkpoints/action_search_manager_20260508"
LOG_ROOT="$ARTIFACT_ROOT/logs/action_search_manager_20260508"
SESSION_NAME="${ACTION_SEARCH_MANAGER_SESSION:-action_search_manager_20260508}"
QUEUE_FILE="${ACTION_SEARCH_MANAGER_QUEUE:-$SCRIPT_DIR/action_search_manager_20260508_queue.tsv}"
ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
MANAGER_GPU="${ACTION_SEARCH_MANAGER_GPU:-3}"
POLL_SEC="${ACTION_SEARCH_MANAGER_POLL_SEC:-300}"
IDLE_AFTER_DONE_SEC="${ACTION_SEARCH_MANAGER_IDLE_AFTER_DONE_SEC:-1800}"

mkdir -p "$RESULT_ROOT" "$CKPT_ROOT" "$LOG_ROOT"

result_exists() {
  local task="$1"
  local run_name="$2"
  compgen -G "$RESULT_ROOT/$task/$run_name/*/result.json" >/dev/null
}

run_active() {
  local run_name="$1"
  pgrep -af "train_same_env_bcpolicy_probe.py" \
    | grep -F -- "--run-name $run_name" >/dev/null 2>&1
}

gpu_busy() {
  nvidia-smi -i "$MANAGER_GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '[0-9]+'
}

wait_for_gpu_idle() {
  while gpu_busy; do
    echo "[$(date '+%F %T')] gpu=$MANAGER_GPU busy; sleeping ${POLL_SEC}s"
    nvidia-smi -i "$MANAGER_GPU" --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits || true
    sleep "$POLL_SEC"
  done
}

sr_from_result_glob() {
  local task="$1"
  local run_name="$2"
  python - "$RESULT_ROOT" "$task" "$run_name" <<'PY'
import glob
import json
import sys
from pathlib import Path

root, task, run_name = sys.argv[1:4]
paths = sorted(glob.glob(str(Path(root) / task / run_name / "*" / "result.json")))
if not paths:
    print("NA")
    raise SystemExit
with open(paths[-1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
print(payload.get("sr", payload.get("mean_per_env_sr", "NA")))
PY
}

write_status() {
  local status="$RESULT_ROOT/STATUS.md"
  {
    echo "# Action Search Manager Status"
    echo
    echo "- time: $(date '+%F %T')"
    echo "- gpu: $MANAGER_GPU"
    echo "- queue: $QUEUE_FILE"
    echo
    echo "## Replay ceiling"
    if [ -s "$ARTIFACT_ROOT/results/action_replay_ceiling_20260508/_summary.tsv" ]; then
      tail -n 16 "$ARTIFACT_ROOT/results/action_replay_ceiling_20260508/_summary.tsv"
    else
      echo "missing"
    fi
    echo
    echo "## Manager queue"
    while IFS=$'\t' read -r enabled priority task cfg run_name epochs batch_size eval_episodes note; do
      [[ "${enabled:-}" =~ ^# ]] && continue
      [ "${enabled:-0}" = "1" ] || continue
      if result_exists "$task" "$run_name"; then
        echo "- done priority=$priority task=$task run=$run_name sr=$(sr_from_result_glob "$task" "$run_name") note=$note"
      elif run_active "$run_name"; then
        echo "- running priority=$priority task=$task run=$run_name note=$note"
      else
        echo "- pending priority=$priority task=$task run=$run_name note=$note"
      fi
    done < "$QUEUE_FILE"
    echo
    echo "## GPU"
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits || true
  } > "$status"
}

prepare_env() {
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/prepare_live_eval_env.sh"
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/cpu_limit_env.sh"
  # shellcheck disable=SC1091
  source /home/cvlab-dgx/anaconda3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"
  export CUDA_VISIBLE_DEVICES="$MANAGER_GPU"
  export ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
  export ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
}

run_train() {
  local task="$1"
  local cfg="$2"
  local run_name="$3"
  local epochs="$4"
  local batch_size="$5"
  local eval_episodes="$6"

  if result_exists "$task" "$run_name"; then
    echo "skip existing result: task=$task run=$run_name sr=$(sr_from_result_glob "$task" "$run_name")"
    return 0
  fi
  if run_active "$run_name"; then
    echo "skip active run: task=$task run=$run_name"
    return 0
  fi

  wait_for_gpu_idle
  prepare_env

  local python_bin="/home/cvlab-dgx/anaconda3/envs/${ENV_NAME}/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin="$(command -v python)"
  fi

  local cmd=(
    "$python_bin" "$REPO_ROOT/scripts/train_same_env_bcpolicy_probe.py"
    --task "$task"
    --dataset-config-path "$REPO_ROOT/$cfg"
    --epochs "$epochs"
    --batch-size "$batch_size"
    --num-workers 1
    --eval-episodes "$eval_episodes"
    --device cuda:0
    --seed 0
    --run-name "$run_name"
    --output-root "$RESULT_ROOT"
    --checkpoint-root "$CKPT_ROOT"
  )

  echo "=== MANAGER TRAIN START gpu=$MANAGER_GPU task=$task run=$run_name cfg=$cfg epochs=$epochs $(date '+%F %T') ==="
  elsa_run_with_cpu_limit "$MANAGER_GPU" "${cmd[@]}"
  echo "=== MANAGER TRAIN END gpu=$MANAGER_GPU task=$task run=$run_name sr=$(sr_from_result_glob "$task" "$run_name") $(date '+%F %T') ==="
}

manager_loop() {
  cd "$REPO_ROOT"
  exec > >(tee -a "$LOG_ROOT/manager.log") 2>&1
  echo "=== ACTION SEARCH MANAGER START gpu=$MANAGER_GPU $(date '+%F %T') ==="
  echo "queue: $QUEUE_FILE"

  while true; do
    local launched_any=0
    write_status
    while IFS=$'\t' read -r enabled priority task cfg run_name epochs batch_size eval_episodes note; do
      [[ "${enabled:-}" =~ ^# ]] && continue
      [ "${enabled:-0}" = "1" ] || continue
      if result_exists "$task" "$run_name"; then
        echo "already done priority=$priority task=$task run=$run_name sr=$(sr_from_result_glob "$task" "$run_name")"
        continue
      fi
      echo "next priority=$priority task=$task run=$run_name note=$note"
      run_train "$task" "$cfg" "$run_name" "$epochs" "$batch_size" "$eval_episodes"
      launched_any=1
      write_status
      break
    done < "$QUEUE_FILE"

    if [ "$launched_any" = "0" ]; then
      echo "[$(date '+%F %T')] queue exhausted; sleeping ${IDLE_AFTER_DONE_SEC}s"
      sleep "$IDLE_AFTER_DONE_SEC"
    fi
  done
}

launch() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "session already exists: $SESSION_NAME"
    return 0
  }
  tmux new-session -d -s "$SESSION_NAME" -n manager \
    "cd '$REPO_ROOT' && bash '$0' --worker"
  echo "started manager: $SESSION_NAME"
  echo "logs: $LOG_ROOT/manager.log"
}

case "${1:-}" in
  --worker)
    manager_loop
    ;;
  *)
    launch
    ;;
esac
