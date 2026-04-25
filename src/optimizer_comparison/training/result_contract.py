"""Training result contract helpers."""

from __future__ import annotations

from pathlib import Path


TrainingResult = dict[str, object]


# /**
#  * Создает пустую структуру артефактов для training-result.
#  *
#  * @return Словарь с обязательными ключами артефактов, включая будущие пути модели и токенизатора.
#  */
def build_empty_artifacts() -> dict[str, object]:
    return {
        "run_dir": None,
        "config_path": None,
        "result_path": None,
        "model": {
            "local_path": None,
            "hf_repo_id": None,
            "hf_commit_url": None,
            "upload_status": "not_applicable",
            "upload_error": None,
        },
        "tokenizer": {
            "local_path": None,
        },
        "checkpoints": {
            "local_path": None,
        },
    }


# /**
#  * Создает training-result в едином формате для mock и настоящего training.
#  *
#  * @param run_name Имя эксперимента или запуска.
#  * @param status Статус выполнения training-функции.
#  * @param final_loss Итоговый loss, если он известен.
#  * @param training_time_seconds Время обучения в секундах.
#  * @param max_memory_mb Максимальная использованная память в мегабайтах.
#  * @return Словарь training-result с вложенными metrics и artifacts.
#  */
def build_training_result(
    run_name: str,
    status: str,
    final_loss: float | None,
    training_time_seconds: float | None,
    max_memory_mb: float | None,
) -> TrainingResult:
    return {
        "status": status,
        "run_name": run_name,
        "metrics": {
            "final_loss": final_loss,
            "training_time_seconds": training_time_seconds,
            "max_memory_mb": max_memory_mb,
        },
        "artifacts": build_empty_artifacts(),
    }


# /**
#  * Записывает локальные пути run/config/result в training-result.
#  *
#  * @param result Training-result, который нужно дополнить путями артефактов.
#  * @param run_dir Директория конкретного запуска.
#  * @param config_path Путь к сохраненному resolved config.
#  * @param result_path Путь, по которому будет сохранен result.json.
#  * @return Тот же training-result с обновленной секцией artifacts.
#  */
def set_local_artifact_paths(
    result: TrainingResult,
    run_dir: str | Path,
    config_path: str | Path,
    result_path: str | Path,
) -> TrainingResult:
    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    artifacts["run_dir"] = str(run_dir)
    artifacts["config_path"] = str(config_path)
    artifacts["result_path"] = str(result_path)
    return result
