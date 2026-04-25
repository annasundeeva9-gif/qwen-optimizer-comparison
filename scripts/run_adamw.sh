#!/usr/bin/env bash
set -euo pipefail

python -m optimizer_comparison.train mode=full model=qwen_0_5b optimizer=adamw experiment=adamw_baseline
