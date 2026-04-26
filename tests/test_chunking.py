from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from optimizer_comparison.data.chunking import (
    chunk_tokenized_dataset,
    chunk_tokenized_splits,
    get_eos_token_id,
    load_or_chunk_splits,
)
from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    save_dataset_local,
)


# /**
#  * Тестовый токенизатор с фиксированным EOS-токеном.
#  */
class FakeTokenizer:
    eos_token_id = 99


# /**
#  * Создает минимальный data-конфиг для тестов chunking-а.
#  *
#  * @param final_dir Локальная директория final dataset.
#  * @param max_length Длина финального чанка.
#  * @param save_local Флаг сохранения final dataset.
#  * @param force_reload Флаг пересборки final dataset вместо загрузки с диска.
#  * @return Hydra-like data-конфиг.
#  */
def make_data_config(
    final_dir: Path,
    max_length: int = 4,
    save_local: bool = True,
    force_reload: bool = False,
) -> Any:
    return OmegaConf.create(
        {
            "tokenization": {
                "max_length": max_length,
            },
            "final": {
                "dir": str(final_dir),
                "save_local": save_local,
                "force_reload": force_reload,
            },
        }
    )


# /**
#  * Создает tokenized split для тестов chunking-а.
#  *
#  * @return DatasetDict с tokenized train и validation.
#  */
def make_tokenized_splits() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": [[1, 2], [3, 4, 5]],
                    "attention_mask": [[1, 1], [1, 1, 1]],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "input_ids": [[10, 11], [12, 13]],
                    "attention_mask": [[1, 1], [1, 1]],
                }
            ),
        }
    )


# /**
#  * Проверяет чтение eos_token_id из токенизатора.
#  *
#  * @return None.
#  */
def test_get_eos_token_id_returns_tokenizer_value() -> None:
    assert get_eos_token_id(FakeTokenizer()) == 99


# /**
#  * Проверяет понятную ошибку, если токенизатор не содержит eos_token_id.
#  *
#  * @return None.
#  */
def test_get_eos_token_id_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="eos_token_id"):
        get_eos_token_id(object())


# /**
#  * Проверяет EOS concat, полные чанки и отбрасывание короткого хвоста.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_chunk_tokenized_dataset_adds_eos_and_drops_remainder(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5]]})
    config = make_data_config(final_dir=tmp_path / "final", max_length=4)

    chunked_dataset = chunk_tokenized_dataset(
        dataset=dataset,
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert chunked_dataset["input_ids"] == [[1, 2, 99, 3]]
    assert chunked_dataset["attention_mask"] == [[1, 1, 1, 1]]
    assert "labels" not in chunked_dataset.column_names


# /**
#  * Проверяет независимую обработку train и validation split-ов.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_chunk_tokenized_splits_keeps_splits_independent(tmp_path: Path) -> None:
    config = make_data_config(final_dir=tmp_path / "final", max_length=3)

    final_splits = chunk_tokenized_splits(
        tokenized_splits=make_tokenized_splits(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert set(final_splits.keys()) == {"train", "validation"}
    assert final_splits["train"]["input_ids"] == [[1, 2, 99], [3, 4, 5]]
    assert final_splits["validation"]["input_ids"] == [[10, 11, 99], [12, 13, 99]]


# /**
#  * Проверяет, что некорректный max_length отклоняется явно.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_chunk_tokenized_dataset_rejects_non_positive_max_length(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"input_ids": [[1, 2, 3]]})
    config = make_data_config(final_dir=tmp_path / "final", max_length=0)

    with pytest.raises(ValueError, match="max_length"):
        chunk_tokenized_dataset(dataset=dataset, tokenizer=FakeTokenizer(), config=config)


# /**
#  * Проверяет сохранение final dataset при save_local=true.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_chunk_splits_saves_when_enabled(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    config = make_data_config(final_dir=final_dir, save_local=True)

    final_splits = load_or_chunk_splits(
        tokenized_splits=make_tokenized_splits(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert dataset_local_path_exists(final_dir)
    assert set(final_splits.keys()) == {"train", "validation"}


# /**
#  * Проверяет загрузку final dataset с диска при наличии копии и force_reload=false.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_chunk_splits_loads_local_copy(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    saved_splits = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}),
            "validation": Dataset.from_dict(
                {"input_ids": [[4, 5, 6]], "attention_mask": [[1, 1, 1]]}
            ),
        }
    )
    save_dataset_local(dataset=saved_splits, path=final_dir)
    config = make_data_config(final_dir=final_dir, force_reload=False)

    final_splits = load_or_chunk_splits(
        tokenized_splits=make_tokenized_splits(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert final_splits["train"]["input_ids"] == [[1, 2, 3]]
    assert final_splits["validation"]["input_ids"] == [[4, 5, 6]]


# /**
#  * Проверяет, что force_reload=true пересобирает final dataset при наличии копии.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_chunk_splits_force_reload_rebuilds_final_dataset(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    saved_splits = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}),
            "validation": Dataset.from_dict(
                {"input_ids": [[4, 5, 6]], "attention_mask": [[1, 1, 1]]}
            ),
        }
    )
    save_dataset_local(dataset=saved_splits, path=final_dir)
    config = make_data_config(final_dir=final_dir, max_length=3, force_reload=True)

    final_splits = load_or_chunk_splits(
        tokenized_splits=make_tokenized_splits(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert final_splits["train"]["input_ids"] == [[1, 2, 99], [3, 4, 5]]
