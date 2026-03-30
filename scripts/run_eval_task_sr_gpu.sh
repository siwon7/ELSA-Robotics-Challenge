#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <task> <round> [local_epochs]"
  exit 1
fi

"$(cd "$(dirname "$0")" && pwd)/run_fk_eval_one.sh" "$@"
