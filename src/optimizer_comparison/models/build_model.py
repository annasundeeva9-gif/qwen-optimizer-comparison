"""Model loading helpers."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM


# /**
#  * Преобразует строковое значение dtype из конфига в torch dtype.
#  *
#  * @param dtype_name Значение model.torch_dtype из Hydra-конфига.
#  * @return torch dtype или None, если параметр не нужно передавать в HuggingFace.
#  */
def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None

    normalized_dtype = str(dtype_name).lower()
    if normalized_dtype == "auto":
        return None

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_by_name.get(normalized_dtype)
    if dtype is None:
        raise ValueError(f"Unsupported model.torch_dtype: {dtype_name}")
    return dtype


# /**
#  * Создает causal language model по Hydra-конфигу.
#  *
#  * @param config Конфигурация модели с именем checkpoint-а и параметрами загрузки.
#  * @return HuggingFace causal language model.
#  */
def build_model(config: DictConfig) -> Any:
    model_name = config.get("pretrained_name_or_path", None)
    if model_name is None:
        raise ValueError("model.pretrained_name_or_path must be set for model loading.")

    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    
    dtype = resolve_torch_dtype(config.get("torch_dtype", None))
    if dtype is not None:
        kwargs["dtype"] = dtype

    return AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
