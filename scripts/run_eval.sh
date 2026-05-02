#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [extra Hydra overrides...]" >&2
  exit 1
fi

RUN_DIR="$1"
shift

echo "Starting evaluation for run: ${RUN_DIR}"
python -m optimizer_comparison.evaluate \
  mode=full \
  model=qwen_0_5b \
  "evaluation.source.run_dir=${RUN_DIR}" \
  "$@"
