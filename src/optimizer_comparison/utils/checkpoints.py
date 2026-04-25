"""Checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path


# /**
#  * Создает директорию, если она отсутствует.
#  *
#  * @param path Путь к директории.
#  * @return Тот же путь после создания директории.
#  */
def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
