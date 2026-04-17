#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/ensure_local_xvfb.sh"

export ELSA_SIM_HEADLESS="${ELSA_SIM_HEADLESS:-0}"
export ELSA_SIM_RENDERER="${ELSA_SIM_RENDERER:-opengl}"
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri}"

if [ -r /lib/x86_64-linux-gnu/libffi.so.7 ]; then
  export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
unset QT_QPA_PLATFORM
