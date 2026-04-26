"""Hydra entrypoint for preparing the training dataset."""

from __future__ import annotations

from typing import Any

import hydra
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig

from optimizer_comparison.data.chunking import load_or_chunk_splits
from optimizer_comparison.data.dataset_loader import load_or_download_raw_dataset
from optimizer_comparison.data.preprocessing import load_or_preprocess_dataset
from optimizer_comparison.data.splits import load_or_build_splits
from optimizer_comparison.data.tokenization import build_tokenizer, load_or_tokenize_splits


# /**
#  * Возвращает количество строк в Dataset или DatasetDict.
#  *
#  * @param dataset Dataset или DatasetDict.
#  * @return Количество строк по каждому split-у или для одного dataset.
#  */
def collect_dataset_sizes(dataset: Dataset | DatasetDict) -> int | dict[str, int]:
    if isinstance(dataset, DatasetDict):
        return {split_name: len(split_dataset) for split_name, split_dataset in dataset.items()}
    return len(dataset)


# /**
#  * Ограничивает raw dataset для smoke-режима, если лимит задан в mode-конфиге.
#  *
#  * @param dataset Raw Dataset до preprocessing и split.
#  * @param config Полный Hydra-конфиг проекта с mode-секцией.
#  * @return Исходный Dataset или его первый поднабор заданного размера.
#  */
def apply_smoke_sample_limit(dataset: Dataset, config: DictConfig) -> Dataset:
    mode_config = config.get("mode", {})
    smoke_samples_limit = mode_config.get("smoke_samples_limit", None)
    if smoke_samples_limit is None:
        return dataset

    limit = int(smoke_samples_limit)
    if limit <= 0:
        raise ValueError("mode.smoke_samples_limit must be positive or null.")

    return dataset.select(range(min(limit, len(dataset))))


# /**
#  * Запускает полный data pipeline в фиксированном порядке.
#  *
#  * @param config Полный Hydra-конфиг проекта.
#  * @return Metadata с путями и размерами промежуточных представлений датасета.
#  */
def run_data_pipeline(config: DictConfig) -> dict[str, Any]:
    data_config = config.data

    raw_dataset = load_or_download_raw_dataset(data_config)
    raw_dataset = apply_smoke_sample_limit(raw_dataset, config)
    preprocessed_dataset = load_or_preprocess_dataset(raw_dataset, data_config)
    split_dataset = load_or_build_splits(preprocessed_dataset, data_config)
    tokenizer = build_tokenizer(config.model)
    tokenized_dataset = load_or_tokenize_splits(split_dataset, tokenizer, data_config)
    final_dataset = load_or_chunk_splits(tokenized_dataset, tokenizer, data_config)

    return {
        "raw": {
            "path": str(data_config.raw.dir),
            "size": collect_dataset_sizes(raw_dataset),
        },
        "preprocessing": {
            "path": str(data_config.preprocessing.dir),
            "size": collect_dataset_sizes(preprocessed_dataset),
        },
        "split": {
            "path": str(data_config.split.dir),
            "size": collect_dataset_sizes(split_dataset),
            "validation_size": float(data_config.split.validation_size),
            "seed": int(data_config.split.seed),
        },
        "tokenization": {
            "path": str(data_config.tokenization.dir),
            "size": collect_dataset_sizes(tokenized_dataset),
            "max_length": int(data_config.tokenization.max_length),
            "min_length_tokens": int(data_config.tokenization.min_length_tokens),
            "num_proc": bool(data_config.tokenization.num_proc),
        },
        "final": {
            "path": str(data_config.final.dir),
            "size": collect_dataset_sizes(final_dataset),
        },
    }


# /**
#  * Запускает подготовку данных через Hydra.
#  *
#  * @param config Полный Hydra-конфиг проекта.
#  * @return None.
#  */
@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(config: DictConfig) -> None:
    run_data_pipeline(config)


if __name__ == "__main__":
    main()
