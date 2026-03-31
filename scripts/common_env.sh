#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ELSA_ROOT="${ELSA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export ELSA_ENV_NAME="${ELSA_ENV_NAME:-clip310}"
export ELSA_VENV_PATH="${ELSA_VENV_PATH:-$ELSA_ROOT/.venv}"
export CONDA_BASE="${CONDA_BASE:-/home/cv7/miniconda3}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/cv7/tools/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu20_04}"
export ELSA_RLBENCH_ROOT="${ELSA_RLBENCH_ROOT:-/home/cv7/siwon/hdp/rlbench}"
export ELSA_XVFB_DISPLAY="${ELSA_XVFB_DISPLAY:-:98}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="$ELSA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export LIBGL_ALWAYS_SOFTWARE=1
