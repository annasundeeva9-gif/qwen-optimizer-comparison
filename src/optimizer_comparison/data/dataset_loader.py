"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from datasets import (  # type: ignore[import-untyped]
    load_dataset as hf_load_dataset,
)
from datasets import (
    load_from_disk,
)
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import resolve_project_path


class LocalDataset(Protocol):
    # /**
    #  * Сохраняет dataset в формате HuggingFace Datasets.
    #  *
    #  * @param dataset_path Целевая директория.
    #  * @return None.
    #  */
    def save_to_disk(self, dataset_path: str) -> None: ...

# /**
#  * Проверяет, похожа ли директория на сохраненный локальный датасет.
#  *
#  * @param path Абсолютный путь или путь относительно корня проекта.
#  * @return True, если директория существует и содержит файлы.
#  */
def dataset_local_path_exists(path: str | Path) -> bool:
    dataset_path = resolve_project_path(path)
    return dataset_path.is_dir() and any(dataset_path.iterdir())


# /**
#  * Сохраняет HuggingFace Dataset или DatasetDict в локальную директорию.
#  *
#  * @param dataset Датасет с методом save_to_disk.
#  * @param path Абсолютный путь или путь относительно корня проекта.
#  * @return Абсолютный путь к директории сохраненного датасета.
#  */
def save_dataset_local(dataset: LocalDataset, path: str | Path) -> Path:
    dataset_path = resolve_project_path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_path))
    return dataset_path


# /**
#  * Загружает локально сохраненный HuggingFace Dataset или DatasetDict.
#  *
#  * @param path Абсолютный путь или путь относительно корня проекта.
#  * @return Объект датасета, восстановленный через HuggingFace load_from_disk.
#  */
def load_dataset_local(path: str | Path) -> Any:
    dataset_path = resolve_project_path(path)
    return load_from_disk(str(dataset_path))


# /**
#  * Возвращает идентификатор исходного HuggingFace dataset.
#  *
#  * @param config Конфигурация данных с секцией raw.
#  * @return Идентификатор dataset repo для datasets.load_dataset.
#  */
def get_hf_dataset_id(config: DictConfig) -> str:
    hf_dataset_id = config.raw.get("hf_dataset_id", None)
    if hf_dataset_id is None:
        raise ValueError("data.raw.hf_dataset_id must be set for raw dataset loading.")
    return str(hf_dataset_id)


# /**
#  * Возвращает имя split-а исходного HuggingFace dataset.
#  *
#  * @param config Конфигурация данных с секцией raw.
#  * @return Имя HuggingFace split-а для datasets.load_dataset.
#  */
def get_hf_split(config: DictConfig) -> str:
    hf_split = config.raw.get("hf_split", None)
    if hf_split is None:
        raise ValueError("data.raw.hf_split must be set for raw dataset loading.")
    return str(hf_split)


# /**
#  * Загружает raw-датасет из HuggingFace Datasets без локального кэша проекта.
#  *
#  * @param config Конфигурация данных с секцией raw.
#  * @return Raw dataset из указанного HuggingFace split-а.
#  */
def load_raw_dataset(config: DictConfig) -> Any:
    dataset_id = get_hf_dataset_id(config)
    hf_split = get_hf_split(config)

    return hf_load_dataset(dataset_id, split=hf_split)


# /**
#  * Загружает raw-датасет из локальной директории или скачивает его из HuggingFace.
#  *
#  * @param config Конфигурация данных, включая источник и режим локального хранения.
#  * @return Raw dataset без изменений содержимого.
#  */
def load_or_download_raw_dataset(config: DictConfig) -> Any:
    raw_dir = str(config.raw.dir)
    force_reload = bool(config.raw.force_reload)

    if dataset_local_path_exists(raw_dir) and not force_reload:
        return load_dataset_local(raw_dir)

    dataset = load_raw_dataset(config)
    if bool(config.raw.save_local):
        save_dataset_local(dataset=dataset, path=raw_dir)
    return dataset


# /**
#  * Загружает raw-датасет на основе секции data из Hydra-конфига.
#  *
#  * @param config Конфигурация данных, включая источник и режим локального хранения.
#  * @return Raw dataset без изменений содержимого.
#  */
def load_dataset_from_config(config: DictConfig) -> Any:
    return load_or_download_raw_dataset(config)
