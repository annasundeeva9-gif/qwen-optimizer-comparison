#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

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

HF_ARGS=()
if [[ -n "${HF_REPO_ID}" ]]; then
  HF_ARGS=("--hf-repo-id" "${HF_REPO_ID}")
fi

bash scripts/main/train_eval.sh \
  --mode smoke \
  --model tiny_qwen_2_5 \
  --optimizer adamw \
  --experiment smoke_adamw_tiny \
  "${HF_ARGS[@]}" \
  data.final.dir=outputs/datasets/final/openwebtext_100k_smoke \
  "${EXTRA_OVERRIDES[@]}"
