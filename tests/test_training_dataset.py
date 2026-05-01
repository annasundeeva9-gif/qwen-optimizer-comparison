from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from optimizer_comparison.data.dataset_loader import save_dataset_local
from optimizer_comparison.training.trainer import (
    load_final_training_dataset,
    validate_final_training_dataset,
    validate_final_training_split,
)


# /**
#  * Создает минимальный data-конфиг с final dataset path.
#  *
#  * @param final_dir Локальная директория final dataset.
#  * @return Hydra-like data-конфиг.
#  */
def make_data_config(final_dir: Path) -> Any:
    return OmegaConf.create(
        {
            "final": {
                "dir": str(final_dir),
            },
        }
    )


# /**
#  * Создает валидный final DatasetDict для training loop.
#  *
#  * @return DatasetDict с train и validation split-ами.
#  */
def make_final_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "input_ids": [[4, 5, 6]],
                    "attention_mask": [[1, 1, 1]],
                }
            ),
        }
    )


# /**
#  * Проверяет успешную загрузку final DatasetDict с диска.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_final_training_dataset_loads_saved_dataset_dict(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    save_dataset_local(dataset=make_final_dataset(), path=final_dir)

    dataset = load_final_training_dataset(make_data_config(final_dir))

    assert set(dataset.keys()) == {"train", "validation"}
    assert dataset["train"]["input_ids"] == [[1, 2, 3]]
    assert dataset["validation"]["attention_mask"] == [[1, 1, 1]]


# /**
#  * Проверяет понятную ошибку, если final dataset директория отсутствует.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_final_training_dataset_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Final training dataset directory"):
        load_final_training_dataset(make_data_config(tmp_path / "missing"))


# /**
#  * Проверяет понятную ошибку, если на диске сохранен Dataset вместо DatasetDict.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_final_training_dataset_rejects_plain_dataset(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    save_dataset_local(
        dataset=Dataset.from_dict({"input_ids": [[1]], "attention_mask": [[1]]}),
        path=final_dir,
    )

    with pytest.raises(TypeError, match="DatasetDict"):
        load_final_training_dataset(make_data_config(final_dir))


# /**
#  * Проверяет ошибку при отсутствии обязательного split-а.
#  *
#  * @return None.
#  */
def test_validate_final_training_dataset_requires_train_and_validation() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                }
            )
        }
    )

    with pytest.raises(ValueError, match="validation"):
        validate_final_training_dataset(dataset)


# /**
#  * Проверяет ошибку при отсутствии обязательной колонки.
#  *
#  * @return None.
#  */
def test_validate_final_training_split_requires_input_ids_and_attention_mask() -> None:
    dataset = Dataset.from_dict({"input_ids": [[1, 2, 3]]})

    with pytest.raises(ValueError, match="attention_mask"):
        validate_final_training_split(dataset=dataset, split_name="train")
