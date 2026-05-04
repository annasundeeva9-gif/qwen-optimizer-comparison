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
    def save_to_disk(self, dataset_path: str) -> None:
        """Saves the dataset in Hugging Face Datasets format."""
        ...


def dataset_local_path_exists(path: str | Path) -> bool:
    """Checks whether a path looks like a saved local dataset."""
    dataset_path = resolve_project_path(path)
    return dataset_path.is_dir() and any(dataset_path.iterdir())


def save_dataset_local(dataset: LocalDataset, path: str | Path) -> Path:
    """Saves a Hugging Face Dataset or DatasetDict into a local directory."""
    dataset_path = resolve_project_path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_path))
    return dataset_path


def load_dataset_local(path: str | Path) -> Any:
    """Loads a locally saved Hugging Face Dataset or DatasetDict."""
    dataset_path = resolve_project_path(path)
    return load_from_disk(str(dataset_path))


def get_hf_dataset_id(config: DictConfig) -> str:
    """Returns the source Hugging Face dataset id."""
    hf_dataset_id = config.raw.get("hf_dataset_id", None)
    if hf_dataset_id is None:
        raise ValueError("data.raw.hf_dataset_id must be set for raw dataset loading.")
    return str(hf_dataset_id)


def get_hf_split(config: DictConfig) -> str:
    """Returns the source Hugging Face dataset split name."""
    hf_split = config.raw.get("hf_split", None)
    if hf_split is None:
        raise ValueError("data.raw.hf_split must be set for raw dataset loading.")
    return str(hf_split)


def load_raw_dataset(config: DictConfig) -> Any:
    """Loads the raw dataset from Hugging Face Datasets."""
    dataset_id = get_hf_dataset_id(config)
    hf_split = get_hf_split(config)

    return hf_load_dataset(dataset_id, split=hf_split)


def load_or_download_raw_dataset(config: DictConfig) -> Any:
    """Loads the raw dataset locally or downloads it from Hugging Face."""
    raw_dir = str(config.raw.dir)
    force_reload = bool(config.raw.force_reload)

    if dataset_local_path_exists(raw_dir) and not force_reload:
        return load_dataset_local(raw_dir)

    dataset = load_raw_dataset(config)
    if bool(config.raw.save_local):
        save_dataset_local(dataset=dataset, path=raw_dir)
    return dataset


def load_dataset_from_config(config: DictConfig) -> Any:
    """Loads the raw dataset from the Hydra data config."""
    return load_or_download_raw_dataset(config)
