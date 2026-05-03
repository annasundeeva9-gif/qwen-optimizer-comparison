"""lm-evaluation-harness result parser."""

from __future__ import annotations

import csv
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
#  * Возвращает true для основной evaluation-метрики без stderr и служебных полей.
#  *
#  * @param metric_name Имя metric из lm-evaluation-harness.
#  * @param metric_value Значение metric из lm-evaluation-harness.
#  * @return True, если metric нужно логировать в MLflow как основное число.
#  */
def is_primary_lm_eval_metric(metric_name: str, metric_value: object) -> bool:
    base_name = metric_name.split(",", maxsplit=1)[0]
    return is_scalar_metric(metric_value) and not base_name.endswith("_stderr")


# /**
#  * Извлекает основные scalar-метрики из секции results JSON-а lm-evaluation-harness.
#  *
#  * @param results Секция results из JSON-а.
#  * @return Flat-словарь основных scalar metrics для MLflow.
#  */
def flatten_lm_eval_metrics(results: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task_name, task_result in results.items():
        if not isinstance(task_result, dict):
            continue
        for metric_name, metric_value in task_result.items():
            if is_primary_lm_eval_metric(metric_name=str(metric_name), metric_value=metric_value):
                metrics[build_metric_name(task_name=task_name, metric_name=str(metric_name))] = (
                    float(metric_value)
                )
    return metrics


# /**
#  * Возвращает имя stderr-метрики для основной lm-evaluation-harness metric.
#  *
#  * @param metric_name Имя основной metric, например acc,none.
#  * @return Имя stderr metric, например acc_stderr,none.
#  */
def build_stderr_metric_name(metric_name: str) -> str:
    if "," not in metric_name:
        return f"{metric_name}_stderr"
    base_name, suffix = metric_name.split(",", maxsplit=1)
    return f"{base_name}_stderr,{suffix}"


# /**
#  * Собирает строки компактной CSV-таблицы из results JSON-а lm-evaluation-harness.
#  *
#  * @param results Секция results из JSON-а.
#  * @return Список строк с task, metric, value и stderr.
#  */
def build_lm_eval_summary_rows(results: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_name, task_result in results.items():
        if not isinstance(task_result, dict):
            continue

        alias = task_result.get("alias", task_name)
        for metric_name, metric_value in task_result.items():
            if not is_primary_lm_eval_metric(
                metric_name=str(metric_name),
                metric_value=metric_value,
            ):
                continue

            stderr_name = build_stderr_metric_name(str(metric_name))
            stderr_value = task_result.get(stderr_name, "")
            rows.append(
                {
                    "task": str(task_name),
                    "alias": str(alias),
                    "metric": str(metric_name),
                    "value": str(float(metric_value)),
                    "stderr": str(float(stderr_value)) if is_scalar_metric(stderr_value) else "",
                }
            )
    return rows


# /**
#  * Записывает компактную CSV-таблицу evaluation results рядом с artifacts запуска.
#  *
#  * @param result_path Путь к JSON-файлу lm-evaluation-harness.
#  * @param output_path Путь к CSV-файлу или None для evaluation_summary.csv рядом с JSON.
#  * @return Путь к записанной CSV-таблице.
#  */
def write_lm_eval_summary_csv(
    result_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    parsed = parse_lm_eval_results(result_path)
    csv_path = Path(output_path) if output_path is not None else Path(result_path).with_name(
        "evaluation_summary.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = parsed["summary_rows"]
    if not isinstance(rows, list):
        raise TypeError("Parsed lm-evaluation-harness summary rows must be a list.")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["task", "alias", "metric", "value", "stderr"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


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
        "summary_rows": build_lm_eval_summary_rows(results),
        "tasks": list(results.keys()),
        "versions": data.get("versions", {}),
        "n-shot": data.get("n-shot", {}),
        "n-samples": data.get("n-samples", {}),
        "higher_is_better": data.get("higher_is_better", {}),
        "config": data.get("config", {}),
    }
