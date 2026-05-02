#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

RUN_DIRS=(
  "outputs/runs/adamw_baseline__REPLACE_ME"
  "outputs/runs/muon_baseline__REPLACE_ME"
)

for run_dir in "${RUN_DIRS[@]}"; do
  if [[ "${run_dir}" == *"REPLACE_ME"* ]]; then
    echo "Edit scripts/workflows/eval_grid.sh and replace placeholder run_dir: ${run_dir}" >&2
    exit 1
  fi

  echo "Starting evaluation grid run: ${run_dir}"
  bash scripts/main/eval.sh "${run_dir}"
done
