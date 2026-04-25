"""Mock training loop for dry runs."""

from __future__ import annotations

from omegaconf import DictConfig


# /**
#  * Возвращает стабильный mock-результат без запуска обучения.
#  *
#  * @param config Полная конфигурация mock-запуска.
#  * @return Словарь с mock-метриками для проверки пайплайна логирования.
#  */
def run_mock_training(config: DictConfig) -> dict[str, object]:
    return {
        "status": "mock_completed",
        "loss": 0.0,
        "training_time_seconds": 0.0,
        "max_memory_mb": 0.0,
        "config_name": str(config.get("experiment", {}).get("name", "unknown")),
    }
