"""Dataset loading skeleton."""

from __future__ import annotations

from omegaconf import DictConfig


# /**
#  * Загружает датасет на основе секции data из Hydra-конфига.
#  *
#  * @param config Конфигурация данных, включая источник и режим локального хранения.
#  * @return Объект датасета. Конкретный тип будет зафиксирован при реализации data pipeline.
#  */
def load_dataset_from_config(config: DictConfig) -> object:
    raise NotImplementedError("Dataset loading implementation is pending.")
