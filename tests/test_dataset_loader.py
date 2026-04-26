from pathlib import Path
from typing import Any

from datasets import Dataset
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from optimizer_comparison.data import dataset_loader
from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    get_hf_dataset_id,
    get_hf_split,
    load_dataset_local,
    load_or_download_raw_dataset,
    load_raw_dataset,
    save_dataset_local,
)


# /**
#  * Создает минимальный конфиг data-секции для тестов raw loader-а.
#  *
#  * @param raw_dir Локальная директория raw dataset.
#  * @param force_reload Флаг принудительной загрузки из источника.
#  * @param save_local Флаг сохранения загруженного датасета на диск.
#  * @return Hydra-like конфигурация данных.
#  */
def make_data_config(
    raw_dir: Path,
    force_reload: bool = False,
    save_local: bool = True,
) -> Any:
    return OmegaConf.create(
        {
            "raw": {
                "hf_dataset_id": "unit-test/source",
                "hf_split": "train",
                "text_column": "text",
                "dir": str(raw_dir),
                "save_local": save_local,
                "force_reload": force_reload,
            },
        }
    )


# /**
#  * Проверяет сохранение и загрузку маленького HuggingFace Dataset.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_save_and_load_dataset_local_roundtrip(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["first", "second"], "id": [1, 2]})
    dataset_path = tmp_path / "raw_dataset"

    saved_path = save_dataset_local(dataset=dataset, path=dataset_path)
    loaded_dataset = load_dataset_local(dataset_path)

    assert saved_path == dataset_path
    assert dataset_local_path_exists(dataset_path)
    assert loaded_dataset.column_names == ["text", "id"]
    assert loaded_dataset["text"] == ["first", "second"]
    assert loaded_dataset["id"] == [1, 2]


# /**
#  * Проверяет чтение идентификатора HuggingFace dataset и HuggingFace split-а.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_get_hf_dataset_id_and_split_read_config(tmp_path: Path) -> None:
    current_config = make_data_config(raw_dir=tmp_path / "current")

    assert get_hf_dataset_id(current_config) == "unit-test/source"
    assert get_hf_split(current_config) == "train"


# /**
#  * Проверяет, что raw loader передает repo id и split в HuggingFace loader.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @param monkeypatch Инструмент pytest для подмены загрузчика.
#  * @return None.
#  */
def test_load_raw_dataset_uses_source_name_and_split(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(name: str, split: str) -> Dataset:
        calls.append((name, split))
        return Dataset.from_dict({"text": ["raw"]})

    monkeypatch.setattr(dataset_loader, "hf_load_dataset", fake_load_dataset)

    dataset = load_raw_dataset(make_data_config(raw_dir=tmp_path / "raw"))

    assert calls == [("unit-test/source", "train")]
    assert dataset["text"] == ["raw"]


# /**
#  * Проверяет, что при наличии локальной raw-копии загрузка из сети не вызывается.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @param monkeypatch Инструмент pytest для подмены загрузчика.
#  * @return None.
#  */
def test_load_or_download_raw_dataset_reuses_local_copy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    local_dataset = Dataset.from_dict({"text": ["local"], "id": [1]})
    save_dataset_local(dataset=local_dataset, path=raw_dir)

    def fail_if_called(name: str, split: str) -> Dataset:
        raise AssertionError(f"Unexpected source loading: {name=} {split=}")

    monkeypatch.setattr(dataset_loader, "hf_load_dataset", fail_if_called)

    loaded_dataset = load_or_download_raw_dataset(make_data_config(raw_dir=raw_dir))

    assert loaded_dataset["text"] == ["local"]
    assert loaded_dataset["id"] == [1]


# /**
#  * Проверяет, что loader скачивает raw dataset, если локальной копии еще нет.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @param monkeypatch Инструмент pytest для подмены загрузчика.
#  * @return None.
#  */
def test_load_or_download_raw_dataset_loads_source_when_local_copy_is_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "missing_raw"
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(name: str, split: str) -> Dataset:
        calls.append((name, split))
        return Dataset.from_dict({"text": ["downloaded"], "id": [7]})

    monkeypatch.setattr(dataset_loader, "hf_load_dataset", fake_load_dataset)

    loaded_dataset = load_or_download_raw_dataset(make_data_config(raw_dir=raw_dir))

    assert calls == [("unit-test/source", "train")]
    assert dataset_local_path_exists(raw_dir)
    assert loaded_dataset.column_names == ["text", "id"]
    assert loaded_dataset["text"] == ["downloaded"]
    assert loaded_dataset["id"] == [7]


# /**
#  * Проверяет, что force_reload игнорирует существующую локальную raw-копию.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @param monkeypatch Инструмент pytest для подмены загрузчика.
#  * @return None.
#  */
def test_load_or_download_raw_dataset_force_reload_uses_source(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    save_dataset_local(dataset=Dataset.from_dict({"text": ["old"]}), path=raw_dir)

    def fake_load_dataset(name: str, split: str) -> Dataset:
        return Dataset.from_dict({"text": ["fresh"]})

    monkeypatch.setattr(dataset_loader, "hf_load_dataset", fake_load_dataset)

    loaded_dataset = load_or_download_raw_dataset(
        make_data_config(raw_dir=raw_dir, force_reload=True, save_local=False)
    )

    assert loaded_dataset["text"] == ["fresh"]


# /**
#  * Проверяет, что force_reload может сохранить новую raw-копию поверх существующей директории.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @param monkeypatch Инструмент pytest для подмены загрузчика.
#  * @return None.
#  */
def test_load_or_download_raw_dataset_force_reload_updates_local_copy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    save_dataset_local(dataset=Dataset.from_dict({"text": ["old"]}), path=raw_dir)

    def fake_load_dataset(name: str, split: str) -> Dataset:
        return Dataset.from_dict({"text": ["fresh"]})

    monkeypatch.setattr(dataset_loader, "hf_load_dataset", fake_load_dataset)

    loaded_dataset = load_or_download_raw_dataset(
        make_data_config(raw_dir=raw_dir, force_reload=True)
    )
    reloaded_dataset = load_dataset_local(raw_dir)

    assert loaded_dataset["text"] == ["fresh"]
    assert reloaded_dataset["text"] == ["fresh"]
