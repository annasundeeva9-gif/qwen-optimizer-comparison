#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODE="full"
MODEL="qwen_0_5b"
TRAINING=""
OPTIMIZER="adamw"
EXPERIMENT=""
EXPERIMENT_CONFIG=""
HF_REPO_ID=""
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --training=*)
      TRAINING="${1#*=}"
      shift
      ;;
    --training)
      TRAINING="$2"
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

OPTIMIZER="$(echo "${OPTIMIZER}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${TRAINING}" ]]; then
  TRAINING="${MODE}"
fi
if [[ "${TRAINING}" == "mock" ]]; then
  TRAINING="smoke"
fi

if [[ "${MODE}" == "mock" ]]; then
  EXPERIMENT_CONFIG="mock_adamw"
elif [[ "${MODE}" == "smoke" ]]; then
  EXPERIMENT_CONFIG="smoke_adamw_tiny"
else
  EXPERIMENT_CONFIG="${OPTIMIZER}_baseline"
fi

if [[ -z "${EXPERIMENT}" ]]; then
  EXPERIMENT="${EXPERIMENT_CONFIG}"
fi

HF_OVERRIDES=()
if [[ -n "${HF_REPO_ID}" ]]; then
  HF_OVERRIDES=(
    "artifacts.hf_hub.use=true"
    "artifacts.hf_hub.repo_id=${HF_REPO_ID}"
  )
fi

echo "Starting training: mode=${MODE} model=${MODEL} optimizer=${OPTIMIZER} experiment=${EXPERIMENT}"
python -m optimizer_comparison.train \
  "mode=${MODE}" \
  "model=${MODEL}" \
  "training=${TRAINING}" \
  "optimizer=${OPTIMIZER}" \
  "experiment=${EXPERIMENT_CONFIG}" \
  "experiment.name=${EXPERIMENT}" \
  "${HF_OVERRIDES[@]}" \
  "${EXTRA_OVERRIDES[@]}"
