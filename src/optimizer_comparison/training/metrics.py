"""Placeholder helpers for future report metrics."""

from __future__ import annotations


# /**
#  * Заглушка будущего сборщика дополнительных метрик для отчетов.
#  *
#  * Основной train-пайплайн уже сохраняет базовые метрики через Trainer и MLflow.
#  * Этот модуль нужен только для будущего расширенного анализа.
#  *
#  * @return Словарь с числовыми метриками обучения.
#  */
def collect_training_metrics() -> dict[str, float]:
    raise NotImplementedError(
        "Extra report metrics placeholder is not used by the training pipeline."
    )
