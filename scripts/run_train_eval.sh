#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${MODE:-smoke}"
MODEL="${MODEL:-tiny_qwen_2_5}"
OPTIMIZER="${OPTIMIZER:-adamw}"
EXPERIMENT="${EXPERIMENT:-smoke_adamw_tiny}"

DATA_OVERRIDES=()
if [[ "${MODE}" == "smoke" ]]; then
  DATA_OVERRIDES=("data.final.dir=outputs/datasets/final/openwebtext_100k_smoke")
fi

echo "Starting training: mode=${MODE} model=${MODEL} optimizer=${OPTIMIZER} experiment=${EXPERIMENT}"
python -m optimizer_comparison.train \
  "mode=${MODE}" \
  "model=${MODEL}" \
  "optimizer=${OPTIMIZER}" \
  "experiment=${EXPERIMENT}" \
  "${DATA_OVERRIDES[@]}" \
  "$@"

RUN_DIR="$(ls -td "outputs/runs/${EXPERIMENT}__"* | head -n 1)"

echo "Starting evaluation for run: ${RUN_DIR}"
python -m optimizer_comparison.evaluate \
  "mode=${MODE}" \
  "model=${MODEL}" \
  "evaluation.source.run_dir=${RUN_DIR}"
