"""MLflow logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import resolve_project_path
from optimizer_comparison.training.result_contract import TrainingResult


# /**
#  * Проверяет, включен ли MLflow tracking в полном Hydra-конфиге.
#  *
#  * @param config Полная конфигурация запуска.
#  * @return True, если tracking включен.
#  */
def is_tracking_enabled(config: DictConfig) -> bool:
    return bool(config.get("tracking", {}).get("enabled", False))


# /**
#  * Настраивает MLflow для текущего запуска.
#  *
#  * @param config Полная конфигурация запуска.
#  * @return None.
#  */
def setup_mlflow(config: DictConfig) -> None:
    tracking_config = config.tracking
    tracking_uri = resolve_project_path(str(tracking_config.tracking_uri))

    mlflow.set_tracking_uri(tracking_uri.as_uri())
    mlflow.set_experiment(str(tracking_config.experiment_name))


# /**
#  * Готовит компактный набор параметров training-запуска для MLflow.
#  *
#  * @param config Полная конфигурация запуска.
#  * @return Словарь параметров, пригодных для MLflow log_params.
#  */
def collect_training_params(config: DictConfig) -> dict[str, str]:
    params = {
        "project.name": config.get("project", {}).get("name"),
        "mode.name": config.get("mode", {}).get("name"),
        "model.name": config.get("model", {}).get("name"),
        "model.pretrained_name_or_path": config.get("model", {}).get("pretrained_name_or_path"),
        "optimizer.name": config.get("optimizer", {}).get("name"),
        "experiment.name": config.get("experiment", {}).get("name"),
        "data.split.seed": config.get("data", {}).get("split", {}).get("seed"),
        "training.num_train_epochs": config.get("training", {}).get("num_train_epochs"),
        "training.per_device_train_batch_size": config.get("training", {}).get(
            "per_device_train_batch_size"
        ),
        "training.gradient_accumulation_steps": config.get("training", {}).get(
            "gradient_accumulation_steps"
        ),
        "training.max_steps": config.get("training", {}).get("max_steps"),
    }

    return {key: str(value) for key, value in params.items() if value is not None}


# /**
#  * Логирует параметры training-запуска в текущий MLflow run.
#  *
#  * @param config Полная конфигурация запуска.
#  * @return None.
#  */
def log_run_config(config: DictConfig) -> None:
    mlflow.log_params(collect_training_params(config))


# /**
#  * Логирует минимальные training-метрики в текущий MLflow run.
#  *
#  * @param training_result Training-result с секцией metrics.
#  * @return None.
#  */
def log_training_metrics(training_result: TrainingResult) -> None:
    metrics = training_result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("Training result metrics must be a dictionary.")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, int | float):
            mlflow.log_metric(f"train/{metric_name}", float(metric_value))


# /**
#  * Логирует локальные артефакты training-запуска в текущий MLflow run.
#  *
#  * @param config Полная конфигурация запуска.
#  * @param training_result Training-result с путями к локальным артефактам.
#  * @return None.
#  */
def log_training_artifacts(config: DictConfig, training_result: TrainingResult) -> None:
    if not bool(config.get("tracking", {}).get("log_artifacts", False)):
        return

    artifacts = training_result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    for artifact_key in ("config_path", "result_path"):
        artifact_path = artifacts.get(artifact_key)
        if artifact_path is not None and Path(str(artifact_path)).is_file():
            mlflow.log_artifact(str(artifact_path))


# /**
#  * Логирует полный training run в MLflow, если tracking включен.
#  *
#  * @param config Полная конфигурация запуска.
#  * @param training_result Training-result с метриками и путями к артефактам.
#  * @return None.
#  */
def log_training_run(config: DictConfig, training_result: TrainingResult) -> None:
    if not is_tracking_enabled(config):
        return

    setup_mlflow(config)

    run_name = str(training_result.get("run_name", config.get("experiment", {}).get("name", "run")))
    with mlflow.start_run(run_name=run_name):
        log_run_config(config)
        mlflow.set_tag("run.status", str(training_result.get("status", "unknown")))

        experiment_tags = config.get("experiment", {}).get("tags", {})
        if isinstance(experiment_tags, dict):
            mlflow.set_tags({str(key): str(value) for key, value in experiment_tags.items()})

        log_training_metrics(training_result)
        log_training_artifacts(config=config, training_result=training_result)


# /**
#  * Логирует расширенные evaluation-метрики в текущий MLflow run.
#  *
#  * @param evaluation_result Evaluation-result будущего evaluation-пайплайна.
#  * @return None.
#  */
def log_evaluation_metrics(evaluation_result: dict[str, Any]) -> None:
    raise NotImplementedError("Evaluation metrics logging is pending evaluation pipeline.")


# /**
#  * Логирует артефакты evaluation-запуска в текущий MLflow run.
#  *
#  * @param evaluation_result Evaluation-result будущего evaluation-пайплайна.
#  * @return None.
#  */
def log_evaluation_artifacts(evaluation_result: dict[str, Any]) -> None:
    raise NotImplementedError("Evaluation artifact logging is pending evaluation pipeline.")


# /**
#  * Логирует полный evaluation run в MLflow.
#  *
#  * @param config Полная конфигурация запуска.
#  * @param evaluation_result Evaluation-result будущего evaluation-пайплайна.
#  * @return None.
#  */
def log_evaluation_run(config: DictConfig, evaluation_result: dict[str, Any]) -> None:
    raise NotImplementedError("Evaluation run logging is pending evaluation pipeline.")
