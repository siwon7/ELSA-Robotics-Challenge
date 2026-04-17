#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ELSA_ROOT="${ELSA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export ELSA_ENV_NAME="${ELSA_ENV_NAME:-elsa_challenge}"
export CONDA_BASE="${CONDA_BASE:-/home/cvlab-dgx/anaconda3}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/cvlab-dgx/siwon/CoppeliaSim_Player_V4_1_0_Ubuntu20_04}"
export ELSA_RLBENCH_ROOT="${ELSA_RLBENCH_ROOT:-/home/cvlab-dgx/siwon/object_centric_diffusion/third_party/RLBench}"
export ELSA_COLOSSEUM_ROOT="${ELSA_COLOSSEUM_ROOT:-/tmp/robot-colosseum}"
export ELSA_XVFB_DISPLAY="${ELSA_XVFB_DISPLAY:-:98}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="$ELSA_ROOT:$ELSA_COLOSSEUM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export LIBGL_ALWAYS_SOFTWARE=1
