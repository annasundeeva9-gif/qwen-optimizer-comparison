"""Model builder skeleton."""

from __future__ import annotations

from omegaconf import DictConfig


# /**
#  * Создает токенизатор по Hydra-конфигу.
#  *
#  * @param config Конфигурация модели.
#  * @return Токенизатор. Конкретный тип будет зависеть от HuggingFace.
#  */
def build_tokenizer(config: DictConfig) -> object:
    raise NotImplementedError("Tokenizer builder implementation is pending.")


# /**
#  * Создает модель по Hydra-конфигу.
#  *
#  * @param config Конфигурация модели.
#  * @return Модель. Конкретный тип будет зависеть от HuggingFace.
#  */
def build_model(config: DictConfig) -> object:
    raise NotImplementedError("Model builder implementation is pending.")
