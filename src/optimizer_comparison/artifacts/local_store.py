"""Local artifact storage helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# /**
#  * Возвращает корневую директорию проекта.
#  *
#  * @return Абсолютный путь к корню репозитория.
#  */
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# /**
#  * Преобразует путь относительно проекта в абсолютный путь.
#  *
#  * @param path Абсолютный путь или путь относительно корня проекта.
#  * @return Абсолютный путь.
#  */
def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return get_project_root() / candidate


# /**
#  * Создает локальную директорию конкретного запуска.
#  *
#  * @param artifacts_root_dir Базовая директория локальных артефактов из Hydra-конфига.
#  * @param experiment_name Имя эксперимента из Hydra-конфига.
#  * @param timestamp Опциональный timestamp для тестов и воспроизводимых путей.
#  * @return Абсолютный путь к созданной директории запуска.
#  */
def create_run_dir(
    artifacts_root_dir: str | Path,
    experiment_name: str,
    timestamp: str | None = None,
) -> Path:
    run_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = resolve_project_path(artifacts_root_dir) / experiment_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# /**
#  * Сохраняет полностью resolved Hydra-конфиг в директорию запуска.
#  *
#  * @param config Полная конфигурация запуска.
#  * @param run_dir Директория конкретного запуска.
#  * @param filename Имя YAML-файла для сохранения конфига.
#  * @return Путь к сохраненному YAML-файлу.
#  */
def save_resolved_config(
    config: DictConfig,
    run_dir: str | Path,
    filename: str = "config.yaml",
) -> Path:
    output_path = Path(run_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_path, resolve=True)
    return output_path


# /**
#  * Сохраняет словарь в JSON-файл внутри локального хранилища артефактов.
#  *
#  * @param data Данные для сохранения.
#  * @param output_path Путь к JSON-файлу.
#  * @return Путь к сохраненному JSON-файлу.
#  */
def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
