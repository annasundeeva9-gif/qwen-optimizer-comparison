#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

HF_REPO_ID="${HF_REPO_ID:-}"
OPTIMIZER="adamw"
EXPERIMENT=""
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --optimizer=*|--optim=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --optimizer|--optim)
      OPTIMIZER="$2"
      shift 2
      ;;
    optimizer=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --experiment=*)
      EXPERIMENT="${1#*=}"
      shift
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    experiment.name=*)
      EXPERIMENT="${1#*=}"
      shift
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
if [[ -z "${EXPERIMENT}" ]]; then
  EXPERIMENT="smoke_${OPTIMIZER}_tiny"
fi

HF_ARGS=()
if [[ -n "${HF_REPO_ID}" ]]; then
  HF_ARGS=("--hf-repo-id" "${HF_REPO_ID}")
fi

bash scripts/main/train_eval.sh \
  --mode smoke \
  --model tiny_qwen_2_5 \
  --optimizer "${OPTIMIZER}" \
  --experiment "${EXPERIMENT}" \
  "${HF_ARGS[@]}" \
  data.final.dir=outputs/datasets/final/openwebtext_100k_smoke \
  "experiment.tags.optimizer=${OPTIMIZER}" \
  "${EXTRA_OVERRIDES[@]}"
