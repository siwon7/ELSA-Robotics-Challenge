#!/usr/bin/env bash

set -euo pipefail

LIMIT_W="${1:-180}"
GPUS="${GPUS:-0 1 2 3}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

echo "Applying NVIDIA GPU power limit: ${LIMIT_W}W to GPUs: ${GPUS}"
for gpu in $GPUS; do
  sudo nvidia-smi -i "$gpu" -pm 1 || true
  sudo nvidia-smi -i "$gpu" -pl "$LIMIT_W"
done

nvidia-smi --query-gpu=index,power.draw,power.limit --format=csv,noheader,nounits
