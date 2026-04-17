#!/usr/bin/env bash
set -euo pipefail

ELSA_XVFB_CACHE="${ELSA_XVFB_CACHE:-/tmp/xvfb-local}"
ELSA_XVFB_DISPLAY="${ELSA_XVFB_DISPLAY:-:98}"
ELSA_XVFB_LOG="${ELSA_XVFB_LOG:-$ELSA_XVFB_CACHE/xvfb.log}"

display_ready() {
  local display_name="$1"
  if [ -z "$display_name" ]; then
    return 1
  fi
  DISPLAY="$display_name" xdpyinfo >/dev/null 2>&1
}

ensure_xvfb_binary() {
  if command -v Xvfb >/dev/null 2>&1; then
    echo "$(command -v Xvfb)"
    return 0
  fi

  mkdir -p "$ELSA_XVFB_CACHE"
  local xvfb_bin="$ELSA_XVFB_CACHE/usr/bin/Xvfb"
  if [ ! -x "$xvfb_bin" ]; then
    local deb_path
    deb_path="$(find /tmp -maxdepth 1 -name 'xvfb_*_amd64.deb' | head -n 1)"
    if [ -z "$deb_path" ]; then
      (cd /tmp && apt download xvfb >/dev/null)
      deb_path="$(find /tmp -maxdepth 1 -name 'xvfb_*_amd64.deb' | head -n 1)"
    fi
    dpkg -x "$deb_path" "$ELSA_XVFB_CACHE"
  fi
  echo "$xvfb_bin"
}

if display_ready "${DISPLAY:-}"; then
  export DISPLAY
  exit 0
fi

XVFB_BIN="$(ensure_xvfb_binary)"

if ! display_ready "$ELSA_XVFB_DISPLAY"; then
  mkdir -p "$ELSA_XVFB_CACHE"
  nohup "$XVFB_BIN" "$ELSA_XVFB_DISPLAY" -screen 0 1280x1024x24 -ac \
    >"$ELSA_XVFB_LOG" 2>&1 &
  sleep 2
fi

if ! display_ready "$ELSA_XVFB_DISPLAY"; then
  echo "failed to start Xvfb on $ELSA_XVFB_DISPLAY" >&2
  [ -f "$ELSA_XVFB_LOG" ] && tail -n 50 "$ELSA_XVFB_LOG" >&2
  exit 1
fi

export DISPLAY="$ELSA_XVFB_DISPLAY"
