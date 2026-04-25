"""Run metadata helpers."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


# /**
#  * Преобразует Hydra-конфиг в обычный словарь для логирования.
#  *
#  * @param config Конфигурация запуска.
#  * @return Словарь с данными конфигурации.
#  */
def flatten_config(config: DictConfig) -> dict[str, object]:
    return dict(OmegaConf.to_container(config, resolve=True))
