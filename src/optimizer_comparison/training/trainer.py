"""Training loop skeleton."""

from __future__ import annotations

from omegaconf import DictConfig

from optimizer_comparison.training.result_contract import TrainingResult


# /**
#  * Запускает настоящий training loop.
#  *
#  * @param config Полная конфигурация обучения.
#  * @return Training-result в общем формате с метриками и путями к артефактам.
#  */
def run_training(config: DictConfig) -> TrainingResult:
    raise NotImplementedError("Training loop implementation is pending.")
