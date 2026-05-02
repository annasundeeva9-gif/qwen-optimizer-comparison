"""Placeholder entrypoint for future evaluation reports."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


# /**
#  * Заглушка будущего сборщика сравнительных отчетов по evaluation.
#  *
#  * Этот модуль не участвует в запуске train-пайплайна и будет реализован позже,
#  * когда появятся финальные таблицы и графики для отчета.
#  *
#  * @param config Полная конфигурация сравнения, собранная Hydra.
#  * @return None. Таблицы и графики будущей реализации будут сохраняться в outputs/reports.
#  */
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    raise NotImplementedError(
        "Comparison report placeholder is not used by the training pipeline."
    )


if __name__ == "__main__":
    main()
