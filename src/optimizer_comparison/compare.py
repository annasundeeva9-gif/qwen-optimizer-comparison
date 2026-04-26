"""Entrypoint for comparing experiment results."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


# /**
#  * Собирает результаты нескольких запусков в сравнительный отчет.
#  *
#  * @param config Полная конфигурация сравнения, собранная Hydra.
#  * @return None. Таблицы и графики будущей реализации будут сохраняться в outputs/reports.
#  */
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    raise NotImplementedError("Comparison pipeline skeleton is created, implementation is pending.")


if __name__ == "__main__":
    main()
