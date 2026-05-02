"""lm-evaluation-harness result parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# /**
#  * Проверяет, что значение можно логировать как scalar metric.
#  *
#  * @param value Значение из JSON lm-evaluation-harness.
#  * @return True, если значение является числовой scalar-метрикой.
#  */
def is_scalar_metric(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# /**
#  * Нормализует имя task metric для MLflow.
#  *
#  * @param task_name Имя evaluation task.
#  * @param metric_name Имя metric из lm-evaluation-harness.
#  * @return Имя metric с префиксом eval_harness.
#  */
def build_metric_name(task_name: str, metric_name: str) -> str:
    normalized_metric_name = metric_name.replace(",", "/")
    return f"eval_harness/{task_name}/{normalized_metric_name}"


# /**
#  * Извлекает scalar-метрики из секции results JSON-а lm-evaluation-harness.
#  *
#  * @param results Секция results из JSON-а.
#  * @return Flat-словарь scalar metrics для MLflow и отчетов.
#  */
def flatten_lm_eval_metrics(results: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task_name, task_result in results.items():
        if not isinstance(task_result, dict):
            continue
        for metric_name, metric_value in task_result.items():
            if is_scalar_metric(metric_value):
                metrics[build_metric_name(task_name=task_name, metric_name=str(metric_name))] = (
                    float(metric_value)
                )
    return metrics


# /**
#  * Парсит оригинальный JSON результатов lm-evaluation-harness.
#  *
#  * @param result_path Путь к JSON-файлу lm-evaluation-harness.
#  * @return Нормализованный словарь с flat metrics и полезной metadata.
#  */
def parse_lm_eval_results(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path)
    if not path.is_file():
        raise FileNotFoundError(f"lm-evaluation-harness result file does not exist: {result_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", None)
    if not isinstance(results, dict):
        raise ValueError("lm-evaluation-harness result JSON must contain a 'results' object.")

    return {
        "result_path": str(path),
        "metrics": flatten_lm_eval_metrics(results),
        "tasks": list(results.keys()),
        "versions": data.get("versions", {}),
        "n-shot": data.get("n-shot", {}),
        "n-samples": data.get("n-samples", {}),
        "higher_is_better": data.get("higher_is_better", {}),
        "config": data.get("config", {}),
    }
