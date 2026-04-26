from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
    save_dataset_local,
)
from optimizer_comparison.data.splits import build_splits, load_or_build_splits


# /**
#  * Создает минимальный data-конфиг для тестов split-а.
#  *
#  * @param split_raw_dir Локальная директория split raw dataset.
#  * @param seed Seed случайного split-а.
#  * @param save_local Флаг сохранения split-а на диск.
#  * @param force_reload Флаг пересборки split-а вместо загрузки с диска.
#  * @return Hydra-like data-конфиг.
#  */
def make_data_config(
    split_raw_dir: Path,
    seed: int = 42,
    save_local: bool = True,
    force_reload: bool = False,
) -> Any:
    return OmegaConf.create(
        {
            "split": {
                "dir": str(split_raw_dir),
                "validation_size": 0.25,
                "seed": seed,
                "save_local": save_local,
                "force_reload": force_reload,
            },
        }
    )


# /**
#  * Создает synthetic dataset с устойчивым порядком примеров.
#  *
#  * @return Dataset для проверки split-а.
#  */
def make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"text {index}" for index in range(12)],
            "id": list(range(12)),
        }
    )


# /**
#  * Проверяет, что build_splits возвращает train и validation без потери примеров.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_splits_returns_train_and_validation(tmp_path: Path) -> None:
    dataset = make_dataset()
    config = make_data_config(split_raw_dir=tmp_path / "split_raw")

    splits = build_splits(dataset=dataset, config=config)

    assert isinstance(splits, DatasetDict)
    assert set(splits.keys()) == {"train", "validation"}
    assert len(splits["train"]) == 9
    assert len(splits["validation"]) == 3
    assert len(splits["train"]) + len(splits["validation"]) == len(dataset)


# /**
#  * Проверяет воспроизводимость split-а при одинаковом seed.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_splits_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    dataset = make_dataset()
    config = make_data_config(split_raw_dir=tmp_path / "split_raw", seed=123)

    first_split = build_splits(dataset=dataset, config=config)
    second_split = build_splits(dataset=dataset, config=config)

    assert first_split["train"]["id"] == second_split["train"]["id"]
    assert first_split["validation"]["id"] == second_split["validation"]["id"]


# /**
#  * Проверяет, что разные seed меняют случайный split.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_splits_changes_with_different_seed(tmp_path: Path) -> None:
    dataset = make_dataset()
    first_config = make_data_config(split_raw_dir=tmp_path / "first", seed=1)
    second_config = make_data_config(split_raw_dir=tmp_path / "second", seed=2)

    first_split = build_splits(dataset=dataset, config=first_config)
    second_split = build_splits(dataset=dataset, config=second_config)

    assert first_split["validation"]["id"] != second_split["validation"]["id"]


# /**
#  * Проверяет сохранение split-а при save_local=true.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_build_splits_saves_when_enabled(tmp_path: Path) -> None:
    split_raw_dir = tmp_path / "split_raw"
    dataset = make_dataset()
    config = make_data_config(split_raw_dir=split_raw_dir, save_local=True)

    splits = load_or_build_splits(dataset=dataset, config=config)
    loaded_splits = load_dataset_local(split_raw_dir)

    assert dataset_local_path_exists(split_raw_dir)
    assert isinstance(loaded_splits, DatasetDict)
    assert loaded_splits["train"]["id"] == splits["train"]["id"]
    assert loaded_splits["validation"]["id"] == splits["validation"]["id"]


# /**
#  * Проверяет, что split не сохраняется при save_local=false.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_build_splits_does_not_save_when_disabled(tmp_path: Path) -> None:
    split_raw_dir = tmp_path / "split_raw"
    dataset = make_dataset()
    config = make_data_config(split_raw_dir=split_raw_dir, save_local=False)

    load_or_build_splits(dataset=dataset, config=config)

    assert not dataset_local_path_exists(split_raw_dir)


# /**
#  * Проверяет загрузку split-а с диска, если он уже сохранен и force_reload=false.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_build_splits_loads_local_copy(tmp_path: Path) -> None:
    split_raw_dir = tmp_path / "split_raw"
    source_dataset = make_dataset()
    saved_splits = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["saved train"], "id": [100]}),
            "validation": Dataset.from_dict({"text": ["saved validation"], "id": [200]}),
        }
    )
    save_dataset_local(dataset=saved_splits, path=split_raw_dir)
    config = make_data_config(split_raw_dir=split_raw_dir, force_reload=False)

    loaded_splits = load_or_build_splits(dataset=source_dataset, config=config)

    assert loaded_splits["train"]["id"] == [100]
    assert loaded_splits["validation"]["id"] == [200]


# /**
#  * Проверяет, что force_reload=true пересобирает split даже при наличии локальной копии.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_build_splits_force_reload_rebuilds_split(tmp_path: Path) -> None:
    split_raw_dir = tmp_path / "split_raw"
    saved_splits = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["saved train"], "id": [100]}),
            "validation": Dataset.from_dict({"text": ["saved validation"], "id": [200]}),
        }
    )
    save_dataset_local(dataset=saved_splits, path=split_raw_dir)
    config = make_data_config(split_raw_dir=split_raw_dir, force_reload=True, save_local=False)

    rebuilt_splits = load_or_build_splits(dataset=make_dataset(), config=config)

    assert rebuilt_splits["train"]["id"] != [100]
    assert rebuilt_splits["validation"]["id"] != [200]
