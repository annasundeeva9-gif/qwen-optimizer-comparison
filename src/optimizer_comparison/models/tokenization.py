"""Tokenizer loading helpers tied to model configs."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from transformers import AutoTokenizer

TINY_QWEN_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
FULL_QWEN_EOS_TOKEN = "<|endoftext|>"


# /**
#  * Приводит smoke-токенизатор tiny Qwen к EOS-контракту основной Qwen-модели.
#  *
#  * @param tokenizer HuggingFace tokenizer, загруженный из model config.
#  * @param config Конфигурация модели, по которой был загружен tokenizer.
#  * @return Тот же tokenizer с исправленным eos_token для tiny Qwen smoke-модели.
#  */
def align_tiny_qwen_special_tokens(tokenizer: Any, config: DictConfig) -> Any:
    model_name = str(config.get("pretrained_name_or_path", ""))
    tokenizer_name = str(config.get("tokenizer_name_or_path", ""))

    if model_name == TINY_QWEN_MODEL_ID and tokenizer_name == TINY_QWEN_MODEL_ID:
        # Keep smoke chunking compatible with the full Qwen tokenizer contract.
        tokenizer.eos_token = FULL_QWEN_EOS_TOKEN

    return tokenizer


# /**
#  * Создает токенизатор по Hydra-конфигу модели.
#  *
#  * @param config Конфигурация модели с именем токенизатора и флагом trust_remote_code.
#  * @return HuggingFace tokenizer, загруженный через AutoTokenizer.
#  */
def build_tokenizer(config: DictConfig) -> Any:
    tokenizer_name = config.get("tokenizer_name_or_path", None)
    if tokenizer_name is None:
        raise ValueError("model.tokenizer_name_or_path must be set for tokenizer loading.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_name),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    return align_tiny_qwen_special_tokens(tokenizer=tokenizer, config=config)
