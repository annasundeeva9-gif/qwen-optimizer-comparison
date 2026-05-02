#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

HF_REPO_ID="${HF_REPO_ID:-}"

RUNS=(
  "mode=full model=qwen_0_5b optimizer=adamw experiment=adamw_baseline"
  "mode=full model=qwen_0_5b optimizer=muon experiment=muon_baseline"
)

for overrides in "${RUNS[@]}"; do
  echo "Starting training grid run: ${overrides}"
  python -m optimizer_comparison.train ${overrides}
done

if [[ -n "${HF_REPO_ID}" ]]; then
  echo "Uploading MLflow snapshot to Hugging Face: ${HF_REPO_ID}"
  python scripts/upload_mlflow_snapshot.py \
    --mlruns-dir outputs/mlruns \
    --repo-id "${HF_REPO_ID}" \
    --repo-path mlflow/mlruns_after_training_grid.zip
else
  echo "HF_REPO_ID is not set; skipping MLflow snapshot upload." >&2
fi
