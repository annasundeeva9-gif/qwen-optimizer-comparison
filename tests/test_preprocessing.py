from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from optimizer_comparison.data.dataset_loader import dataset_local_path_exists
from optimizer_comparison.data.preprocessing import (
    load_or_preprocess_dataset,
    simple_preprocess,
)


# /**
#  * Создает минимальный data-конфиг для тестов preprocessing-а.
#  *
#  * @param preprocessed_dir Локальная директория preprocessed dataset.
#  * @param save_local Флаг сохранения preprocessed dataset.
#  * @return Hydra-like data-конфиг.
#  */
def make_data_config(
    preprocessed_dir: Path,
    save_local: bool = False,
    force_reload: bool = False,
) -> Any:
    return OmegaConf.create(
        {
            "raw": {
                "text_column": "text",
            },
            "preprocessing": {
                "dir": str(preprocessed_dir),
                "save_local": save_local,
                "force_reload": force_reload,
            },
        }
    )


# /**
#  * Проверяет, что simple_preprocess не меняет корректный raw dataset.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_simple_preprocess_returns_valid_dataset_unchanged(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["first text", "second text"], "id": [1, 2]})
    config = make_data_config(preprocessed_dir=tmp_path / "preprocessed")

    preprocessed = simple_preprocess(dataset=dataset, config=config)

    assert preprocessed is dataset
    assert preprocessed.column_names == ["text", "id"]
    assert preprocessed["text"] == ["first text", "second text"]
    assert preprocessed["id"] == [1, 2]


# /**
#  * Проверяет, что строка из пробелов считается пустой.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_simple_preprocess_rejects_blank_text(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["valid text", "   "]})
    config = make_data_config(preprocessed_dir=tmp_path / "preprocessed")

    with pytest.raises(ValueError, match="empty texts"):
        simple_preprocess(dataset=dataset, config=config)


# /**
#  * Проверяет, что simple_preprocess ищет дубликаты по raw-тексту.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_simple_preprocess_rejects_duplicate_raw_text(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["same text", "same text"]})
    config = make_data_config(preprocessed_dir=tmp_path / "preprocessed")

    with pytest.raises(ValueError, match="duplicate raw texts"):
        simple_preprocess(dataset=dataset, config=config)


# /**
#  * Проверяет, что preprocessed dataset не сохраняется при save_local=false.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_preprocess_dataset_does_not_save_when_disabled(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["valid text"]})
    preprocessed_dir = tmp_path / "preprocessed"
    config = make_data_config(preprocessed_dir=preprocessed_dir, save_local=False)

    preprocessed = load_or_preprocess_dataset(dataset=dataset, config=config)

    assert preprocessed is dataset
    assert not dataset_local_path_exists(preprocessed_dir)


# /**
#  * Проверяет сохранение preprocessed dataset при save_local=true.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_preprocess_dataset_saves_when_enabled(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["valid text"]})
    preprocessed_dir = tmp_path / "preprocessed"
    config = make_data_config(preprocessed_dir=preprocessed_dir, save_local=True)

    load_or_preprocess_dataset(dataset=dataset, config=config)

    assert dataset_local_path_exists(preprocessed_dir)
