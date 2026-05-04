"""Dataset chunking helpers for causal language modeling."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]
from omegaconf import DictConfig

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
    save_dataset_local,
)


def get_eos_token_id(tokenizer: Any) -> int:
    """Returns eos_token_id from the tokenizer."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for chunking.")
    return int(eos_token_id)


def chunk_tokenized_dataset(dataset: Dataset, tokenizer: Any, config: DictConfig) -> Dataset:
    """Concatenates tokenized texts with EOS and splits them into full chunks."""
    if "input_ids" not in dataset.column_names:
        raise ValueError("Tokenized dataset must contain input_ids.")

    max_length = int(config.tokenization.max_length)
    if max_length <= 0:
        raise ValueError("tokenization.max_length must be positive.")

    eos_token_id = get_eos_token_id(tokenizer)
    token_stream: list[int] = []
    for input_ids in dataset["input_ids"]:
        token_stream.extend(int(token_id) for token_id in input_ids)
        token_stream.append(eos_token_id)

    chunk_count = len(token_stream) // max_length
    chunks = [
        token_stream[index * max_length : (index + 1) * max_length]
        for index in range(chunk_count)
    ]

    return Dataset.from_dict(
        {
            "input_ids": chunks,
            "attention_mask": [[1] * max_length for _ in chunks],
        }
    )


def chunk_tokenized_splits(
    tokenized_splits: DatasetDict,
    tokenizer: Any,
    config: DictConfig,
) -> DatasetDict:
    """Applies chunking to train and validation splits independently."""
    return DatasetDict(
        {
            "train": chunk_tokenized_dataset(tokenized_splits["train"], tokenizer, config),
            "validation": chunk_tokenized_dataset(
                tokenized_splits["validation"],
                tokenizer,
                config,
            ),
        }
    )


def load_or_chunk_splits(
    tokenized_splits: DatasetDict,
    tokenizer: Any,
    config: DictConfig,
) -> DatasetDict:
    """Loads the final dataset or builds chunks from tokenized splits."""
    final_dir = str(config.final.dir)

    if dataset_local_path_exists(final_dir) and not bool(config.final.force_reload):
        loaded_dataset = load_dataset_local(final_dir)
        if not isinstance(loaded_dataset, DatasetDict):
            raise TypeError("Final dataset must be a HuggingFace DatasetDict.")
        return loaded_dataset

    final_splits = chunk_tokenized_splits(
        tokenized_splits=tokenized_splits,
        tokenizer=tokenizer,
        config=config,
    )
    if bool(config.final.save_local):
        save_dataset_local(dataset=final_splits, path=final_dir)
    return final_splits
