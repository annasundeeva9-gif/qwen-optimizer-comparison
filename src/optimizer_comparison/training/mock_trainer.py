"""Mock training loop for dry runs."""

from __future__ import annotations

from omegaconf import DictConfig

from optimizer_comparison.training.result_contract import TrainingResult, build_training_result


# /**
#  * Возвращает стабильный mock-результат без запуска обучения.
#  *
#  * @param config Полная конфигурация mock-запуска.
#  * @return Training-result в общем формате с mock-метриками и пустыми артефактами модели.
#  */
def run_mock_training(config: DictConfig) -> TrainingResult:
    run_name = str(config.get("experiment", {}).get("name", "unknown"))

    return build_training_result(
        run_name=run_name,
        status="completed",
        final_loss=0.0,
        training_time_seconds=0.0,
        max_memory_mb=0.0,
    )
