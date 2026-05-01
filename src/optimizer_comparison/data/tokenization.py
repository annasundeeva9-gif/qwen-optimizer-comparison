"""Dataset tokenization helpers."""

from __future__ import annotations

import os
from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
    save_dataset_local,
)


# /**
#  * Преобразует булевый флаг multiprocessing в значение num_proc для Dataset.map.
#  *
#  * @param enabled True, если нужно включить параллельную токенизацию.
#  * @return Количество процессов или None для однопроцессного режима.
#  */
def resolve_num_proc(enabled: bool) -> int | None:
    if not enabled:
        return None
    return max(1, (os.cpu_count() or 1) - 1)


# /**
#  * Токенизирует один split датасета без добавления специальных токенов.
#  *
#  * @param dataset Dataset с raw-текстами.
#  * @param tokenizer Токенизатор HuggingFace или совместимый тестовый stub.
#  * @param config Полная data-секция Hydra-конфига.
#  * @return Dataset с токенизированными полями без исходной текстовой колонки.
#  */
def tokenize_dataset(dataset: Dataset, tokenizer: Any, config: DictConfig) -> Dataset:
    tokenization_config = config.tokenization
    text_column = str(config.raw.text_column)
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset does not contain text column: {text_column}")

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            padding=False,
            truncation=False,
            add_special_tokens=False,
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


# /**
#  * Проверяет, что все примеры имеют минимально допустимую длину в токенах.
#  *
#  * @param dataset Токенизированный Dataset с колонкой input_ids.
#  * @param min_length_tokens Минимальная допустимая длина input_ids.
#  * @return None.
#  */
def validate_min_token_length(dataset: Dataset, min_length_tokens: int) -> None:
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


# /**
#  * Токенизирует train и validation split-ы независимо друг от друга.
#  *
#  * @param splits DatasetDict с ключами train и validation.
#  * @param tokenizer Токенизатор HuggingFace или совместимый тестовый stub.
#  * @param config Полная data-секция Hydra-конфига.
#  * @return DatasetDict с токенизированными train и validation.
#  */
def tokenize_splits(splits: DatasetDict, tokenizer: Any, config: DictConfig) -> DatasetDict:
    return DatasetDict(
        {
            "train": tokenize_dataset(splits["train"], tokenizer, config),
            "validation": tokenize_dataset(splits["validation"], tokenizer, config),
        }
    )


# /**
#  * Загружает tokenized split с диска или токенизирует raw split.
#  *
#  * @param splits DatasetDict с raw train и validation.
#  * @param tokenizer Токенизатор HuggingFace или совместимый тестовый stub.
#  * @param config Полная data-секция Hydra-конфига.
#  * @return DatasetDict с токенизированными split-ами.
#  */
def load_or_tokenize_splits(splits: DatasetDict, tokenizer: Any, config: DictConfig) -> DatasetDict:
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
