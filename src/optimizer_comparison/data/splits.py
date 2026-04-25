"""Dataset split skeleton."""

from __future__ import annotations

from omegaconf import DictConfig


# /**
#  * Создает train/validation split для подготовленного датасета.
#  *
#  * @param dataset Подготовленный датасет.
#  * @param config Конфигурация разбиения данных.
#  * @return Набор split-ов. 
#  */
def build_splits(dataset: object, config: DictConfig) -> object:
    raise NotImplementedError("Dataset split implementation is pending.")
