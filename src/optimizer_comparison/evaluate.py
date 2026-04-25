"""Entrypoint for evaluation runs."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

# /**
#  * Запускает evaluation-пайплайн поверх lm-evaluation-harness.
#  *
#  * @param config Полная конфигурация запуска, собранная Hydra.
#  * @return None. Вывод lm-evaluation-harness должен сохраняться в файл для парсинга.
#  */
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    raise NotImplementedError("Evaluation pipeline skeleton is created, implementation is pending.")


if __name__ == "__main__":
    main()
