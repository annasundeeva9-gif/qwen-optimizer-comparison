"""Dataset preprocessing helpers."""

from __future__ import annotations

from datasets import Dataset  # type: ignore[import-untyped]
from omegaconf import DictConfig

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
    save_dataset_local,
)


def simple_preprocess(dataset: Dataset, config: DictConfig) -> Dataset:
    """Validates the raw dataset without changing its content."""
    text_column = str(config.raw.text_column)
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset does not contain text column: {text_column}")

    texts = dataset[text_column]
    empty_indices = [
        index for index, text in enumerate(texts) if not isinstance(text, str) or not text.strip()
    ]
    if empty_indices:
        raise ValueError(f"Dataset contains empty texts in column '{text_column}'.")

    unique_texts = set(texts)
    if len(unique_texts) != len(texts):
        raise ValueError(f"Dataset contains duplicate raw texts in column '{text_column}'.")

    return dataset


def load_or_preprocess_dataset(dataset: Dataset, config: DictConfig) -> Dataset:
    """Loads a preprocessed dataset or validates the raw dataset."""
    preprocessing_config = config.preprocessing
    preprocessed_dir = str(preprocessing_config.dir)

    if (
        bool(preprocessing_config.save_local)
        and dataset_local_path_exists(preprocessed_dir)
        and not bool(preprocessing_config.force_reload)
    ):
        loaded_dataset = load_dataset_local(preprocessed_dir)
        if not isinstance(loaded_dataset, Dataset):
            raise TypeError("Preprocessed dataset must be a HuggingFace Dataset.")
        return loaded_dataset

    preprocessed_dataset = simple_preprocess(dataset=dataset, config=config)
    if bool(preprocessing_config.save_local):
        save_dataset_local(dataset=preprocessed_dataset, path=preprocessed_dir)
    return preprocessed_dataset
