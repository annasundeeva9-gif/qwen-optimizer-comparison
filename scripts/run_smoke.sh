#!/usr/bin/env bash
set -euo pipefail

python -m optimizer_comparison.train mode=smoke model=tiny optimizer=adamw experiment=smoke_adamw_tiny
