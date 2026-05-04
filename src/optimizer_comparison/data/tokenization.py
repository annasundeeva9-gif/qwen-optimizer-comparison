"""Dataset tokenization helpers."""

from __future__ import annotations

import os
from typing import Any, cast

from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]
from omegaconf import DictConfig

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
    save_dataset_local,
)


def resolve_num_proc(enabled: bool) -> int | None:
    """Converts the multiprocessing flag into a Dataset.map num_proc value."""
    if not enabled:
        return None
    return max(1, (os.cpu_count() or 1) - 1)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, config: DictConfig) -> Dataset:
    """Tokenizes one dataset split without adding special tokens."""
    tokenization_config = config.tokenization
    text_column = str(config.raw.text_column)
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset does not contain text column: {text_column}")

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            tokenizer(
                batch[text_column],
                padding=False,
                truncation=False,
                add_special_tokens=False,
            ),
        )

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": [text_column],
    }
    num_proc = resolve_num_proc(bool(tokenization_config.num_proc))
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    tokenized_dataset = dataset.map(tokenize_batch, **map_kwargs)
    validate_min_token_length(
        dataset=tokenized_dataset,
        min_length_tokens=int(tokenization_config.min_length_tokens),
    )
    return tokenized_dataset


def validate_min_token_length(dataset: Dataset, min_length_tokens: int) -> None:
    """Checks that every example has the minimum allowed token length."""
    if "input_ids" not in dataset.column_names:
        raise ValueError("Tokenized dataset does not contain input_ids.")

    short_indices = [
        index
        for index, input_ids in enumerate(dataset["input_ids"])
        if len(input_ids) < min_length_tokens
    ]
    if short_indices:
        raise ValueError(
            f"Tokenized dataset contains texts shorter than {min_length_tokens} tokens."
        )


def tokenize_splits(splits: DatasetDict, tokenizer: Any, config: DictConfig) -> DatasetDict:
    """Tokenizes train and validation splits independently."""
    return DatasetDict(
        {
            "train": tokenize_dataset(splits["train"], tokenizer, config),
            "validation": tokenize_dataset(splits["validation"], tokenizer, config),
        }
    )


def load_or_tokenize_splits(splits: DatasetDict, tokenizer: Any, config: DictConfig) -> DatasetDict:
    """Loads tokenized splits or tokenizes raw splits."""
    tokenized_dir = str(config.tokenization.dir)

    if dataset_local_path_exists(tokenized_dir) and not bool(config.tokenization.force_reload):
        loaded_dataset = load_dataset_local(tokenized_dir)
        if not isinstance(loaded_dataset, DatasetDict):
            raise TypeError("Tokenized dataset must be a HuggingFace DatasetDict.")
        return loaded_dataset

    tokenized_splits = tokenize_splits(splits=splits, tokenizer=tokenizer, config=config)
    if bool(config.tokenization.save_local):
        save_dataset_local(dataset=tokenized_splits, path=tokenized_dir)
    return tokenized_splits
