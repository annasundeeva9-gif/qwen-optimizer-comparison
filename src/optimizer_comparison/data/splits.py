"""Dataset split helpers."""

from __future__ import annotations

from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]
from omegaconf import DictConfig

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
    save_dataset_local,
)


def build_splits(dataset: Dataset, config: DictConfig) -> DatasetDict:
    """Builds a random train/validation split."""
    split_config = config.split
    split_dataset = dataset.train_test_split(
        test_size=float(split_config.validation_size),
        seed=int(split_config.seed),
        shuffle=True,
    )

    return DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        }
    )


def load_or_build_splits(dataset: Dataset, config: DictConfig) -> DatasetDict:
    """Loads split data or builds it from the preprocessed dataset."""
    split_raw_dir = str(config.split.dir)

    if dataset_local_path_exists(split_raw_dir) and not bool(config.split.force_reload):
        loaded_dataset = load_dataset_local(split_raw_dir)
        if not isinstance(loaded_dataset, DatasetDict):
            raise TypeError("Split raw dataset must be a HuggingFace DatasetDict.")
        return loaded_dataset

    splits = build_splits(dataset=dataset, config=config)
    if bool(config.split.save_local):
        save_dataset_local(dataset=splits, path=split_raw_dir)
    return splits
