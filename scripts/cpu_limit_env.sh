#!/usr/bin/env bash

# Source this from queue launchers before invoking training.  The defaults are
# intentionally conservative because this machine has shown hard-reset behavior
# under concurrent training load.

ELSA_CPU_LIMITS_ENABLED="${ELSA_CPU_LIMITS_ENABLED:-1}"
ELSA_CPU_CORES_PER_GPU="${ELSA_CPU_CORES_PER_GPU:-4}"
ELSA_CPU_THREADS_PER_JOB="${ELSA_CPU_THREADS_PER_JOB:-1}"
ELSA_DATALOADER_WORKERS="${ELSA_DATALOADER_WORKERS:-1}"
ELSA_RUN_LOCK_POLL_SEC="${ELSA_RUN_LOCK_POLL_SEC:-60}"

elsa_export_cpu_limits() {
  local threads="${1:-$ELSA_CPU_THREADS_PER_JOB}"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$threads}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$threads}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$threads}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$threads}"
  export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$threads}"
  export ELSA_TORCH_NUM_THREADS="${ELSA_TORCH_NUM_THREADS:-$threads}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}

elsa_cpu_set_for_gpu() {
  local gpu="$1"
  local override_var="ELSA_CPUSET_GPU${gpu}"
  local override="${!override_var:-}"
  if [ -n "$override" ]; then
    echo "$override"
    return 0
  fi

  local cores_per_gpu="$ELSA_CPU_CORES_PER_GPU"
  case "$cores_per_gpu" in
    ''|*[!0-9]*) cores_per_gpu=4 ;;
  esac
  if [ "$cores_per_gpu" -le 0 ]; then
    echo ""
    return 0
  fi

  local nproc
  nproc="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  case "$nproc" in
    ''|*[!0-9]*) nproc=1 ;;
  esac

  local max_cpu=$((nproc - 1))
  local start=$((gpu * cores_per_gpu))
  while [ "$start" -gt "$max_cpu" ]; do
    start=$((start - cores_per_gpu))
  done
  if [ "$start" -lt 0 ]; then
    start=0
  fi

  local end=$((start + cores_per_gpu - 1))
  if [ "$end" -gt "$max_cpu" ]; then
    end="$max_cpu"
  fi

  echo "${start}-${end}"
}

elsa_run_with_cpu_limit() {
  local gpu="$1"
  shift

  elsa_export_cpu_limits "$ELSA_CPU_THREADS_PER_JOB"

  if [ "$ELSA_CPU_LIMITS_ENABLED" != "1" ]; then
    echo "[cpu-limit] disabled gpu=$gpu"
    "$@"
    return $?
  fi

  local cpu_set
  cpu_set="$(elsa_cpu_set_for_gpu "$gpu")"
  if [ -z "$cpu_set" ]; then
    echo "[cpu-limit] no taskset gpu=$gpu threads=$ELSA_CPU_THREADS_PER_JOB"
    "$@"
    return $?
  fi

  echo "[cpu-limit] gpu=$gpu cpuset=$cpu_set threads=$ELSA_CPU_THREADS_PER_JOB dataloader_workers=$ELSA_DATALOADER_WORKERS"
  if command -v taskset >/dev/null 2>&1; then
    taskset -c "$cpu_set" "$@"
  else
    echo "[cpu-limit] taskset not found; running without affinity pinning"
    "$@"
  fi
}

elsa_existing_run_pids() {
  local run_name="$1"
  pgrep -af "train_same_env_bcpolicy_probe.py" \
    | grep -F -- "--run-name $run_name" \
    | awk '{print $1}' \
    || true
}

elsa_wait_for_existing_run() {
  local run_name="$1"
  local pids

  while true; do
    pids="$(elsa_existing_run_pids "$run_name" | tr '\n' ' ')"
    if [ -z "$pids" ]; then
      return 0
    fi
    echo "[run-lock] waiting for existing run_name=$run_name pids=$pids"
    sleep "$ELSA_RUN_LOCK_POLL_SEC"
  done
}
