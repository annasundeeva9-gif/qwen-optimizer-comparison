from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from optimizer_comparison.data import prepare_data
from optimizer_comparison.data.prepare_data import (
    apply_smoke_sample_limit,
    collect_dataset_sizes,
    run_data_pipeline,
)


# /**
#  * Минимальный tokenizer stub для проверки orchestration.
#  */
class FakeTokenizer:
    pass


# /**
#  * Создает полный Hydra-like конфиг для тестов data entrypoint.
#  *
#  * @return Полный конфиг с data и model секциями.
#  */
def make_config(smoke_samples_limit: int | None = None) -> Any:
    return OmegaConf.create(
        {
            "mode": {
                "smoke_samples_limit": smoke_samples_limit,
            },
            "data": {
                "raw": {"dir": "raw_dir"},
                "preprocessing": {"dir": "preprocessed_dir"},
                "split": {"dir": "split_dir", "validation_size": 0.05, "seed": 42},
                "tokenization": {
                    "dir": "tokenized_dir",
                    "max_length": 1024,
                    "min_length_tokens": 100,
                    "num_proc": False,
                },
                "final": {"dir": "final_dir"},
            },
            "model": {"tokenizer_name_or_path": "unit-test/tokenizer"},
        }
    )


# /**
#  * Проверяет сбор размеров Dataset и DatasetDict.
#  *
#  * @return None.
#  */
def test_collect_dataset_sizes_supports_dataset_and_dataset_dict() -> None:
    dataset = Dataset.from_dict({"text": ["a", "b"]})
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["a", "b"]}),
            "validation": Dataset.from_dict({"text": ["c"]}),
        }
    )

    assert collect_dataset_sizes(dataset) == 2
    assert collect_dataset_sizes(dataset_dict) == {"train": 2, "validation": 1}


# /**
#  * Проверяет, что smoke limit оставляет raw dataset без изменений, если лимит не задан.
#  *
#  * @return None.
#  */
def test_apply_smoke_sample_limit_keeps_dataset_without_limit() -> None:
    dataset = Dataset.from_dict({"text": ["a", "b"]})
    config = make_config(smoke_samples_limit=None)

    limited_dataset = apply_smoke_sample_limit(dataset=dataset, config=config)

    assert limited_dataset is dataset


# /**
#  * Проверяет, что smoke limit берет первые raw-примеры до дальнейшей обработки.
#  *
#  * @return None.
#  */
def test_apply_smoke_sample_limit_selects_first_examples() -> None:
    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "id": [1, 2, 3]})
    config = make_config(smoke_samples_limit=2)

    limited_dataset = apply_smoke_sample_limit(dataset=dataset, config=config)

    assert limited_dataset["text"] == ["a", "b"]
    assert limited_dataset["id"] == [1, 2]


# /**
#  * Проверяет понятную ошибку для некорректного smoke limit.
#  *
#  * @return None.
#  */
def test_apply_smoke_sample_limit_rejects_non_positive_limit() -> None:
    dataset = Dataset.from_dict({"text": ["a"]})
    config = make_config(smoke_samples_limit=0)

    try:
        apply_smoke_sample_limit(dataset=dataset, config=config)
    except ValueError as error:
        assert "smoke_samples_limit" in str(error)
    else:
        raise AssertionError("Expected ValueError for non-positive smoke limit.")


# /**
#  * Проверяет, что верхнеуровневый pipeline вызывает этапы в фиксированном порядке.
#  *
#  * @param monkeypatch Инструмент pytest для подмены этапов pipeline.
#  * @return None.
#  */
def test_run_data_pipeline_calls_steps_in_order(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []
    raw_dataset = Dataset.from_dict({"text": ["raw"]})
    preprocessed_dataset = Dataset.from_dict({"text": ["preprocessed"]})
    split_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["train"]}),
            "validation": Dataset.from_dict({"text": ["validation"]}),
        }
    )
    tokenized_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]]}),
            "validation": Dataset.from_dict({"input_ids": [[4, 5, 6]]}),
        }
    )
    final_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]]}),
            "validation": Dataset.from_dict({"input_ids": [[4, 5, 6]]}),
        }
    )

    def fake_load_raw(config: Any) -> Dataset:
        calls.append("raw")
        return raw_dataset

    def fake_preprocess(dataset: Dataset, config: Any) -> Dataset:
        calls.append("preprocessing")
        assert dataset is raw_dataset
        return preprocessed_dataset

    def fake_split(dataset: Dataset, config: Any) -> DatasetDict:
        calls.append("split")
        assert dataset is preprocessed_dataset
        return split_dataset

    def fake_build_tokenizer(config: Any) -> FakeTokenizer:
        calls.append("tokenizer")
        return FakeTokenizer()

    def fake_tokenize(splits: DatasetDict, tokenizer: FakeTokenizer, config: Any) -> DatasetDict:
        calls.append("tokenization")
        assert splits is split_dataset
        return tokenized_dataset

    def fake_chunk(splits: DatasetDict, tokenizer: FakeTokenizer, config: Any) -> DatasetDict:
        calls.append("final")
        assert splits is tokenized_dataset
        return final_dataset

    monkeypatch.setattr(prepare_data, "load_or_download_raw_dataset", fake_load_raw)
    monkeypatch.setattr(prepare_data, "load_or_preprocess_dataset", fake_preprocess)
    monkeypatch.setattr(prepare_data, "load_or_build_splits", fake_split)
    monkeypatch.setattr(prepare_data, "build_tokenizer", fake_build_tokenizer)
    monkeypatch.setattr(prepare_data, "load_or_tokenize_splits", fake_tokenize)
    monkeypatch.setattr(prepare_data, "load_or_chunk_splits", fake_chunk)

    metadata = run_data_pipeline(make_config())

    assert calls == ["raw", "preprocessing", "split", "tokenizer", "tokenization", "final"]
    assert metadata["raw"]["size"] == 1
    assert metadata["split"]["size"] == {"train": 1, "validation": 1}
    assert metadata["final"]["path"] == "final_dir"


# /**
#  * Проверяет, что верхнеуровневый pipeline применяет smoke limit до preprocessing и split.
#  *
#  * @param monkeypatch Инструмент pytest для подмены этапов pipeline.
#  * @return None.
#  */
def test_run_data_pipeline_applies_smoke_limit_before_preprocessing(
    monkeypatch: MonkeyPatch,
) -> None:
    raw_dataset = Dataset.from_dict({"text": ["raw 1", "raw 2", "raw 3"]})
    split_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["raw 1"]}),
            "validation": Dataset.from_dict({"text": ["raw 2"]}),
        }
    )
    tokenized_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1]]}),
            "validation": Dataset.from_dict({"input_ids": [[2]]}),
        }
    )

    def fake_load_raw(config: Any) -> Dataset:
        return raw_dataset

    def fake_preprocess(dataset: Dataset, config: Any) -> Dataset:
        assert dataset["text"] == ["raw 1", "raw 2"]
        return dataset

    def fake_split(dataset: Dataset, config: Any) -> DatasetDict:
        assert dataset["text"] == ["raw 1", "raw 2"]
        return split_dataset

    monkeypatch.setattr(prepare_data, "load_or_download_raw_dataset", fake_load_raw)
    monkeypatch.setattr(prepare_data, "load_or_preprocess_dataset", fake_preprocess)
    monkeypatch.setattr(prepare_data, "load_or_build_splits", fake_split)
    monkeypatch.setattr(prepare_data, "build_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(prepare_data, "load_or_tokenize_splits", lambda *args: tokenized_dataset)
    monkeypatch.setattr(prepare_data, "load_or_chunk_splits", lambda *args: tokenized_dataset)

    metadata = run_data_pipeline(make_config(smoke_samples_limit=2))

    assert metadata["raw"]["size"] == 2


# /**
#  * Проверяет, что модуль entrypoint импортируется без запуска pipeline.
#  *
#  * @return None.
#  */
def test_prepare_data_module_exposes_main() -> None:
    assert prepare_data.main is not None
