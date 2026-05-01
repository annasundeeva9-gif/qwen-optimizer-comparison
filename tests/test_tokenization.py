from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from optimizer_comparison.data import tokenization
from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    save_dataset_local,
)
from optimizer_comparison.data.tokenization import (
    load_or_tokenize_splits,
    resolve_num_proc,
    tokenize_dataset,
    tokenize_splits,
)
from optimizer_comparison.models import tokenization as model_tokenization
from optimizer_comparison.models.tokenization import (
    FULL_QWEN_EOS_TOKEN,
    TINY_QWEN_MODEL_ID,
    align_tiny_qwen_special_tokens,
    build_tokenizer,
)


# /**
#  * Тестовый токенизатор, который превращает каждое слово в один token id.
#  */
class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    # /**
    #  * Токенизирует batch текстов и запоминает параметры вызова.
    #  *
    #  * @param texts Batch raw-текстов.
    #  * @param padding Флаг padding.
    #  * @param truncation Флаг truncation.
    #  * @param add_special_tokens Флаг автоматического добавления специальных токенов.
    #  * @return Словарь с input_ids и attention_mask.
    #  */
    def __call__(
        self,
        texts: list[str],
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[list[int]]]:
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "truncation": truncation,
                "add_special_tokens": add_special_tokens,
            }
        )
        input_ids = [list(range(len(text.split()))) for text in texts]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(tokens) for tokens in input_ids],
        }


# /**
#  * Минимальный tokenizer stub для проверки правки special tokens.
#  */
class FakeSpecialTokensTokenizer:
    def __init__(self, eos_token: str) -> None:
        self.eos_token = eos_token


# /**
#  * Создает минимальный data-конфиг для тестов токенизации.
#  *
#  * @param tokenized_dir Локальная директория tokenized dataset.
#  * @param min_length_tokens Минимальная допустимая длина tokenized текста.
#  * @param save_local Флаг сохранения tokenized dataset.
#  * @param force_reload Флаг пересборки tokenized dataset вместо загрузки с диска.
#  * @return Hydra-like data-конфиг.
#  */
def make_data_config(
    tokenized_dir: Path,
    min_length_tokens: int = 3,
    save_local: bool = True,
    force_reload: bool = False,
) -> Any:
    return OmegaConf.create(
        {
            "raw": {
                "text_column": "text",
            },
            "tokenization": {
                "dir": str(tokenized_dir),
                "min_length_tokens": min_length_tokens,
                "num_proc": False,
                "save_local": save_local,
                "force_reload": force_reload,
            },
        }
    )


# /**
#  * Создает raw train/validation split для тестов токенизации.
#  *
#  * @return DatasetDict с raw train и validation.
#  */
def make_splits() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": ["one two three", "four five six seven"],
                    "id": [1, 2],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "text": ["eight nine ten"],
                    "id": [3],
                }
            ),
        }
    )


# /**
#  * Проверяет, что токенизатор загружается из model config.
#  *
#  * @param monkeypatch Инструмент pytest для подмены AutoTokenizer.
#  * @return None.
#  */
def test_build_tokenizer_uses_model_config(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    # /**
    #  * Тестовая замена AutoTokenizer.
    #  */
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, trust_remote_code: bool) -> str:
            calls.append((name, trust_remote_code))
            return "tokenizer"

    monkeypatch.setattr(model_tokenization, "AutoTokenizer", FakeAutoTokenizer)
    model_config = OmegaConf.create(
        {
            "tokenizer_name_or_path": "unit-test/tokenizer",
            "trust_remote_code": True,
        }
    )

    loaded_tokenizer = build_tokenizer(model_config)

    assert loaded_tokenizer == "tokenizer"
    assert calls == [("unit-test/tokenizer", True)]


# /**
#  * Проверяет, что tiny Qwen smoke tokenizer получает EOS основной Qwen-модели.
#  *
#  * @return None.
#  */
def test_align_tiny_qwen_special_tokens_uses_full_qwen_eos() -> None:
    tokenizer = FakeSpecialTokensTokenizer(eos_token="<|im_end|>")
    config = OmegaConf.create(
        {
            "pretrained_name_or_path": TINY_QWEN_MODEL_ID,
            "tokenizer_name_or_path": TINY_QWEN_MODEL_ID,
        }
    )

    aligned_tokenizer = align_tiny_qwen_special_tokens(tokenizer=tokenizer, config=config)

    assert aligned_tokenizer is tokenizer
    assert tokenizer.eos_token == FULL_QWEN_EOS_TOKEN


# /**
#  * Проверяет, что special tokens не меняются для остальных моделей.
#  *
#  * @return None.
#  */
def test_align_tiny_qwen_special_tokens_keeps_other_models() -> None:
    tokenizer = FakeSpecialTokensTokenizer(eos_token="<|custom_eos|>")
    config = OmegaConf.create(
        {
            "pretrained_name_or_path": "unit-test/model",
            "tokenizer_name_or_path": "unit-test/tokenizer",
        }
    )

    aligned_tokenizer = align_tiny_qwen_special_tokens(tokenizer=tokenizer, config=config)

    assert aligned_tokenizer is tokenizer
    assert tokenizer.eos_token == "<|custom_eos|>"


# /**
#  * Проверяет преобразование булевого num_proc в параметр Dataset.map.
#  *
#  * @param monkeypatch Инструмент pytest для подмены cpu_count.
#  * @return None.
#  */
def test_resolve_num_proc_uses_boolean_flag(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(tokenization.os, "cpu_count", lambda: 8)

    assert resolve_num_proc(False) is None
    assert resolve_num_proc(True) == 7


# /**
#  * Проверяет токенизацию одного split-а без специальных токенов и без raw text column.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_tokenize_dataset_removes_text_column_and_disables_special_tokens(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["one two three"], "id": [1]})
    tokenizer = FakeTokenizer()
    config = make_data_config(tokenized_dir=tmp_path / "tokenized")

    tokenized_dataset = tokenize_dataset(dataset=dataset, tokenizer=tokenizer, config=config)

    assert "text" not in tokenized_dataset.column_names
    assert tokenized_dataset["id"] == [1]
    assert tokenized_dataset["input_ids"] == [[0, 1, 2]]
    assert tokenizer.calls[0]["padding"] is False
    assert tokenizer.calls[0]["truncation"] is False
    assert tokenizer.calls[0]["add_special_tokens"] is False


# /**
#  * Проверяет, что train и validation токенизуются независимо.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_tokenize_splits_keeps_train_and_validation(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    config = make_data_config(tokenized_dir=tmp_path / "tokenized")

    tokenized_splits = tokenize_splits(splits=make_splits(), tokenizer=tokenizer, config=config)

    assert set(tokenized_splits.keys()) == {"train", "validation"}
    assert tokenized_splits["train"]["id"] == [1, 2]
    assert tokenized_splits["validation"]["id"] == [3]
    assert "text" not in tokenized_splits["train"].column_names
    assert "text" not in tokenized_splits["validation"].column_names


# /**
#  * Проверяет assertion на минимальную длину tokenized текста.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_tokenize_dataset_rejects_short_texts(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"text": ["too short"], "id": [1]})
    tokenizer = FakeTokenizer()
    config = make_data_config(tokenized_dir=tmp_path / "tokenized", min_length_tokens=3)

    with pytest.raises(ValueError, match="shorter than 3 tokens"):
        tokenize_dataset(dataset=dataset, tokenizer=tokenizer, config=config)


# /**
#  * Проверяет сохранение tokenized split-а при save_local=true.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_tokenize_splits_saves_when_enabled(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenizer = FakeTokenizer()
    config = make_data_config(tokenized_dir=tokenized_dir, save_local=True)

    tokenized_splits = load_or_tokenize_splits(
        splits=make_splits(),
        tokenizer=tokenizer,
        config=config,
    )

    assert dataset_local_path_exists(tokenized_dir)
    assert tokenized_splits["train"]["id"] == [1, 2]
    assert tokenized_splits["validation"]["id"] == [3]


# /**
#  * Проверяет загрузку tokenized split-а с диска при наличии копии и force_reload=false.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_tokenize_splits_loads_local_copy(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    saved_splits = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]], "id": [100]}),
            "validation": Dataset.from_dict({"input_ids": [[4, 5, 6]], "id": [200]}),
        }
    )
    save_dataset_local(dataset=saved_splits, path=tokenized_dir)
    config = make_data_config(tokenized_dir=tokenized_dir, force_reload=False)

    tokenized_splits = load_or_tokenize_splits(
        splits=make_splits(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert tokenized_splits["train"]["id"] == [100]
    assert tokenized_splits["validation"]["id"] == [200]


# /**
#  * Проверяет, что force_reload=true пересобирает tokenized split при наличии копии.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_or_tokenize_splits_force_reload_retokenizes(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    saved_splits = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]], "id": [100]}),
            "validation": Dataset.from_dict({"input_ids": [[4, 5, 6]], "id": [200]}),
        }
    )
    save_dataset_local(dataset=saved_splits, path=tokenized_dir)
    config = make_data_config(tokenized_dir=tokenized_dir, force_reload=True, save_local=False)

    tokenized_splits = load_or_tokenize_splits(
        splits=make_splits(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    assert tokenized_splits["train"]["id"] == [1, 2]
    assert tokenized_splits["validation"]["id"] == [3]
