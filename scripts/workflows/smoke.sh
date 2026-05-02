#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

bash scripts/main/train_eval.sh \
  --mode smoke \
  --model tiny_qwen_2_5 \
  --optimizer adamw \
  --experiment smoke_adamw_tiny \
  data.final.dir=outputs/datasets/final/openwebtext_100k_smoke
