#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

HF_REPO_ID="${HF_REPO_ID:-}"
RUN_ID=""
MODE="full"
MODEL="qwen_0_5b"
MLFLOW_REPO_PATH="mlflow/mlruns_after_remote_train.zip"
TOKEN_ENV_VAR="HF_TOKEN"
REVISION=""
SKIP_MLFLOW=false
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
    --run-id=*)
      RUN_ID="${1#*=}"
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --mlflow-repo-path=*)
      MLFLOW_REPO_PATH="${1#*=}"
      shift
      ;;
    --mlflow-repo-path)
      MLFLOW_REPO_PATH="$2"
      shift 2
      ;;
    --token-env-var=*)
      TOKEN_ENV_VAR="${1#*=}"
      shift
      ;;
    --token-env-var)
      TOKEN_ENV_VAR="$2"
      shift 2
      ;;
    --revision=*)
      REVISION="${1#*=}"
      shift
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --skip-mlflow)
      SKIP_MLFLOW=true
      shift
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

if [[ -z "${RUN_ID}" ]]; then
  echo "Run id is required. Pass --run-id <run_id>." >&2
  exit 1
fi

REVISION_ARGS=()
EVAL_REVISION_OVERRIDE=()
if [[ -n "${REVISION}" ]]; then
  REVISION_ARGS=(--revision "${REVISION}")
  EVAL_REVISION_OVERRIDE=("evaluation.source.revision=${REVISION}")
fi

TRACKING_OVERRIDE=()
if [[ "${SKIP_MLFLOW}" == "true" ]]; then
  echo "Skipping MLflow snapshot download and disabling evaluation MLflow logging."
  TRACKING_OVERRIDE=("tracking.enabled=false")
else
  echo "Downloading and merging MLflow snapshot from Hugging Face: ${HF_REPO_ID}/${MLFLOW_REPO_PATH}"
  python scripts/workflows/download_mlflow_snapshot.py \
    --repo-id "${HF_REPO_ID}" \
    --repo-path "${MLFLOW_REPO_PATH}" \
    --mlruns-dir outputs/mlruns \
    --download-dir "outputs/mlflow_downloads/${RUN_ID}" \
    --extract-dir "outputs/mlflow_downloads/${RUN_ID}/extracted" \
    --token-env-var "${TOKEN_ENV_VAR}" \
    "${REVISION_ARGS[@]}"
fi

echo "Downloading run and starting evaluation: ${RUN_ID}"
python -m optimizer_comparison.evaluate \
  "mode=${MODE}" \
  "model=${MODEL}" \
  "evaluation.source.use_hf_hub=true" \
  "evaluation.source.repo_id=${HF_REPO_ID}" \
  "evaluation.source.repo_path=runs/${RUN_ID}" \
  "evaluation.source.download_dir=outputs" \
  "evaluation.source.token_env_var=${TOKEN_ENV_VAR}" \
  "${EVAL_REVISION_OVERRIDE[@]}" \
  "${TRACKING_OVERRIDE[@]}" \
  "${EXTRA_OVERRIDES[@]}"
