"""Checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Creates a directory when it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
