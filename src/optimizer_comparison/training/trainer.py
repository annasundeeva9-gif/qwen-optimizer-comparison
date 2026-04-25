"""Training loop skeleton."""

from __future__ import annotations

from omegaconf import DictConfig


# /**
#  * Запускает настоящий training loop.
#  *
#  * @param config Полная конфигурация обучения.
#  * @return Словарь с первичными результатами обучения и путями к артефактам.
#  */
def run_training(config: DictConfig) -> dict[str, object]:
    raise NotImplementedError("Training loop implementation is pending.")
