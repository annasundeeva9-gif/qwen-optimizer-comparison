#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

HF_REPO_ID="${HF_REPO_ID:-}"
OPTIMIZER="adamw"
EXPERIMENT=""
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
    --optimizer=*|--optim=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --optimizer|--optim)
      OPTIMIZER="$2"
      shift 2
      ;;
    --experiment=*)
      EXPERIMENT="${1#*=}"
      shift
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${HF_REPO_ID}" ]]; then
  echo "HF repo id is required. Pass --hf-repo-id <user/repo> or set HF_REPO_ID." >&2
  exit 1
fi

OPTIMIZER="$(echo "${OPTIMIZER}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${EXPERIMENT}" ]]; then
  EXPERIMENT="${OPTIMIZER}_baseline"
fi

bash scripts/main/train.sh \
  --mode full \
  --model qwen_0_5b \
  "--optimizer=${OPTIMIZER}" \
  "--experiment=${EXPERIMENT}" \
  "--hf-repo-id=${HF_REPO_ID}" \
  "${EXTRA_OVERRIDES[@]}"

echo "Uploading MLflow snapshot to Hugging Face: ${HF_REPO_ID}"
python scripts/workflows/upload_mlflow_snapshot.py \
  --mlruns-dir outputs/mlruns \
  --repo-id "${HF_REPO_ID}" \
  --repo-path mlflow/mlruns_after_remote_train.zip
