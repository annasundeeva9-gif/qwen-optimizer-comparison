#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

HF_REPO_ID="${HF_REPO_ID:-}"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo-id=*)
      HF_REPO_ID="${1#*=}"
      shift
      ;;
    --hf-repo-id)
      HF_REPO_ID="$2"
      shift 2
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

echo "Starting base model Trainer eval_loss run."
python -m optimizer_comparison.evaluate_base_trainer_loss \
  mode=full \
  model=qwen_0_5b \
  training=full \
  optimizer=adamw \
  experiment=adamw_baseline \
  experiment.name=base_qwen_trainer_eval_loss \
  "${EXTRA_OVERRIDES[@]}"

if [[ -n "${HF_REPO_ID}" ]]; then
  echo "Uploading base trainer-eval artifacts to Hugging Face: ${HF_REPO_ID}/runs/__base_qwen_2_5_0_5b__"
  python scripts/workflows/upload_hf_artifact.py \
    --artifact-path outputs/runs/__base_qwen_2_5_0_5b__ \
    --repo-id "${HF_REPO_ID}" \
    --repo-path runs/__base_qwen_2_5_0_5b__

  echo "Uploading MLflow snapshot to Hugging Face: ${HF_REPO_ID}"
  python scripts/workflows/upload_mlflow_snapshot.py \
    --mlruns-dir outputs/mlruns \
    --repo-id "${HF_REPO_ID}" \
    --repo-path mlflow/mlruns_after_base_trainer_eval.zip
fi
