#!/usr/bin/env bash
set -euo pipefail

python -m optimizer_comparison.evaluate mode=full model=qwen_0_5b
