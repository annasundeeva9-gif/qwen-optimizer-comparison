#!/usr/bin/env bash
set -euo pipefail

python -m optimizer_comparison.train mode=mock model=tiny optimizer=adamw experiment=mock_adamw
