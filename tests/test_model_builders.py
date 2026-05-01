from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from optimizer_comparison.models import build_model as model_builder
from optimizer_comparison.models.build_model import build_model, resolve_torch_dtype


# /**
#  * Проверяет поддерживаемые строковые значения torch dtype.
#  *
#  * @return None.
#  */
def test_resolve_torch_dtype_supports_expected_values() -> None:
    assert resolve_torch_dtype(None) is None
    assert resolve_torch_dtype("auto") is None
    assert resolve_torch_dtype("float32") == torch.float32
    assert resolve_torch_dtype("float16") == torch.float16
    assert resolve_torch_dtype("bfloat16") == torch.bfloat16
    assert resolve_torch_dtype("bf16") == torch.bfloat16


# /**
#  * Проверяет понятную ошибку для неподдерживаемого dtype.
#  *
#  * @return None.
#  */
def test_resolve_torch_dtype_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported model.torch_dtype"):
        resolve_torch_dtype("int8")


# /**
#  * Проверяет, что model builder передает HuggingFace ожидаемые параметры загрузки.
#  *
#  * @param monkeypatch Инструмент pytest для подмены AutoModelForCausalLM.
#  * @return None.
#  */
def test_build_model_uses_model_config(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    # /**
    #  * Тестовая замена AutoModelForCausalLM.
    #  */
    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> str:
            calls.append((name, kwargs))
            return "model"

    monkeypatch.setattr(model_builder, "AutoModelForCausalLM", FakeAutoModelForCausalLM)
    model_config = OmegaConf.create(
        {
            "pretrained_name_or_path": "unit-test/model",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
        }
    )

    loaded_model = build_model(model_config)

    assert loaded_model == "model"
    assert calls == [
        (
            "unit-test/model",
            {
                "trust_remote_code": True,
                "dtype": torch.bfloat16,
            },
        )
    ]


# /**
#  * Проверяет, что model builder не передает torch_dtype при null/auto.
#  *
#  * @param monkeypatch Инструмент pytest для подмены AutoModelForCausalLM.
#  * @return None.
#  */
def test_build_model_omits_torch_dtype_when_not_set(monkeypatch: MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    # /**
    #  * Тестовая замена AutoModelForCausalLM.
    #  */
    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> str:
            calls.append(kwargs)
            return name

    monkeypatch.setattr(model_builder, "AutoModelForCausalLM", FakeAutoModelForCausalLM)
    model_config = OmegaConf.create(
        {
            "pretrained_name_or_path": "unit-test/model",
            "trust_remote_code": False,
            "torch_dtype": None,
        }
    )

    build_model(model_config)

    assert calls == [{"trust_remote_code": False}]


# /**
#  * Проверяет понятную ошибку, если в model config нет имени модели.
#  *
#  * @return None.
#  */
def test_build_model_requires_model_name() -> None:
    with pytest.raises(ValueError, match="pretrained_name_or_path"):
        build_model(OmegaConf.create({"torch_dtype": "float32"}))
