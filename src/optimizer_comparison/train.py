"""Entrypoint for training runs."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

# /**
#  * Запускает training-пайплайн с конфигурацией Hydra.
#  *
#  * @param config Полная конфигурация запуска, собранная Hydra.
#  * @return None. Результаты будущей реализации будут сохраняться в outputs и MLflow.
#  */
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    raise NotImplementedError("Training pipeline skeleton is created, implementation is pending.")


if __name__ == "__main__":
    main()
