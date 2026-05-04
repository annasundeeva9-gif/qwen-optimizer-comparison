"""MLflow logging helpers."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import resolve_project_path
from optimizer_comparison.evaluation.result_parser import parse_lm_eval_results
from optimizer_comparison.training.result_contract import TrainingResult

MLFLOW_EXPERIMENT_ID = "0"


def is_tracking_enabled(config: DictConfig) -> bool:
    """Checks whether MLflow tracking is enabled."""
    return bool(config.get("tracking", {}).get("enabled", False))


def setup_mlflow(config: DictConfig) -> None:
    """Configures MLflow for the current run."""
    tracking_config = config.tracking
    tracking_uri = resolve_project_path(str(tracking_config.tracking_uri))

    mlflow.set_tracking_uri(tracking_uri.as_uri())


def collect_training_params(config: DictConfig) -> dict[str, str]:
    """Builds the compact training parameter set for MLflow."""
    params = {
        "project.name": config.get("project", {}).get("name"),
        "mode.name": config.get("mode", {}).get("name"),
        "model.name": config.get("model", {}).get("name"),
        "model.pretrained_name_or_path": config.get("model", {}).get("pretrained_name_or_path"),
        "optimizer.name": config.get("optimizer", {}).get("name"),
        "optimizer.lr": config.get("optimizer", {}).get("lr"),
        "optimizer.weight_decay": config.get("optimizer", {}).get("weight_decay"),
        "optimizer.betas": config.get("optimizer", {}).get("betas"),
        "optimizer.eps": config.get("optimizer", {}).get("eps"),
        "experiment.name": config.get("experiment", {}).get("name"),
        "data.split.seed": config.get("data", {}).get("split", {}).get("seed"),
        "training.seed": config.get("training", {}).get("seed"),
        "training.data_seed": config.get("training", {}).get("data_seed"),
        "training.num_train_epochs": config.get("training", {}).get("num_train_epochs"),
        "training.per_device_train_batch_size": config.get("training", {}).get(
            "per_device_train_batch_size"
        ),
        "training.gradient_accumulation_steps": config.get("training", {}).get(
            "gradient_accumulation_steps"
        ),
        "training.lr_scheduler_type": config.get("training", {}).get("lr_scheduler_type"),
        "training.warmup_ratio": config.get("training", {}).get("warmup_ratio"),
        "training.max_grad_norm": config.get("training", {}).get("max_grad_norm"),
        "training.max_steps": config.get("training", {}).get("max_steps"),
    }

    return {key: str(value) for key, value in params.items() if value is not None}


def log_run_config(config: DictConfig) -> None:
    """Logs training run parameters into the active MLflow run."""
    mlflow.log_params(collect_training_params(config))


def log_training_metrics(training_result: TrainingResult) -> None:
    """Logs top-level training metrics into the active MLflow run."""
    metrics = training_result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("Training result metrics must be a dictionary.")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, int | float):
            mlflow.log_metric(f"train/{metric_name}", float(metric_value))


def log_training_history(training_result: TrainingResult) -> None:
    """Logs step-wise training and evaluation metrics."""
    history = training_result.get("history", [])
    if not isinstance(history, list):
        raise TypeError("Training result history must be a list.")

    metric_names = {
        "loss": "train/loss",
        "learning_rate": "train/learning_rate",
        "grad_norm": "train/grad_norm",
        "eval_loss": "eval/loss",
    }
    for entry in history:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step", None)
        if not isinstance(step, int):
            continue
        for source_name, mlflow_name in metric_names.items():
            value = entry.get(source_name, None)
            if isinstance(value, int | float):
                mlflow.log_metric(mlflow_name, float(value), step=step)


def log_hf_hub_tags(training_result: TrainingResult) -> None:
    """Logs Hugging Face Hub artifact links into MLflow tags."""
    artifacts = training_result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    hf_hub = artifacts.get("hf_hub", {})
    if not isinstance(hf_hub, dict):
        return

    tags = {
        f"hf_hub.{key}": str(hf_hub[key])
        for key in ("repo_id", "commit_url", "revision", "artifact_path")
        if hf_hub.get(key) is not None
    }
    if tags:
        mlflow.set_tags(tags)


def log_custom_artifacts(config: DictConfig, artifact_paths: list[str]) -> None:
    """Logs selected local artifact files into MLflow."""
    if not bool(config.get("tracking", {}).get("log_artifacts", False)):
        return

    for artifact_path in artifact_paths:
        path = resolve_project_path(artifact_path)
        if path.is_file():
            mlflow.log_artifact(str(path))


def log_training_artifacts(config: DictConfig, training_result: TrainingResult) -> None:
    """Logs compact training plot artifacts into MLflow."""
    artifacts = training_result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    plots = artifacts.get("plots", {})
    if not isinstance(plots, dict):
        return

    plot_paths = [
        str(path)
        for path in [plots.get("training_curves_path", None)]
        if path is not None
    ]
    log_custom_artifacts(config=config, artifact_paths=plot_paths)


def log_training_run(config: DictConfig, training_result: TrainingResult) -> str | None:
    """Logs the full training run to MLflow when tracking is enabled."""
    if not is_tracking_enabled(config):
        return None

    setup_mlflow(config)

    run_name = str(training_result.get("run_name", config.get("experiment", {}).get("name", "run")))
    with mlflow.start_run(experiment_id=MLFLOW_EXPERIMENT_ID, run_name=run_name) as active_run:
        log_run_config(config)
        mlflow.set_tag("run.status", str(training_result.get("status", "unknown")))
        mlflow.set_tag("project.run_id", str(training_result.get("run_id", "")))

        experiment_tags = config.get("experiment", {}).get("tags", {})
        if isinstance(experiment_tags, dict):
            mlflow.set_tags({str(key): str(value) for key, value in experiment_tags.items()})

        log_training_metrics(training_result)
        log_training_history(training_result)
        log_hf_hub_tags(training_result)
        log_training_artifacts(config=config, training_result=training_result)

        return str(active_run.info.run_id)


def update_training_run_hf_tags(config: DictConfig, training_result: TrainingResult) -> None:
    """Appends Hugging Face Hub metadata to an existing MLflow training run."""
    if not is_tracking_enabled(config):
        return

    mlflow_run_id = training_result.get("mlflow_run_id", None)
    if mlflow_run_id is None:
        return

    setup_mlflow(config)
    with mlflow.start_run(run_id=str(mlflow_run_id)):
        log_hf_hub_tags(training_result)


def log_evaluation_metrics(parsed_lm_eval_result: dict[str, Any]) -> None:
    """Logs parsed lm-evaluation-harness metrics into MLflow."""
    metrics = parsed_lm_eval_result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("Parsed lm-evaluation-harness metrics must be a dictionary.")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, int | float):
            mlflow.log_metric(str(metric_name), float(metric_value))


def log_evaluation_tags(evaluation_result: dict[str, Any]) -> None:
    """Logs evaluation metadata into MLflow tags."""
    tags = {
        "evaluation.status": str(evaluation_result.get("status", "unknown")),
        "evaluation.result_path": str(evaluation_result.get("lm_eval_result_path", "")),
    }

    model_source = evaluation_result.get("model_source", None)
    if isinstance(model_source, dict):
        source_type = model_source.get("type", None)
        base_model_id = model_source.get("base_model_id", None)
        if source_type is not None:
            tags["evaluation.source_type"] = str(source_type)
        if base_model_id is not None:
            tags["evaluation.base_model_id"] = str(base_model_id)

    raw_log_path = evaluation_result.get("raw_log_path", None)
    if raw_log_path is not None:
        tags["evaluation.raw_log_path"] = str(raw_log_path)

    mlflow.set_tags(tags)


def log_evaluation_artifacts(evaluation_result: dict[str, Any]) -> None:
    """Logs the evaluation summary CSV as an MLflow artifact."""
    summary_path = evaluation_result.get("summary_path", None)
    if summary_path is None:
        return

    path = resolve_project_path(str(summary_path))
    if path.is_file():
        mlflow.log_artifact(str(path))


def log_evaluation_run(config: DictConfig, evaluation_result: dict[str, Any]) -> str | None:
    """Logs the full evaluation run to MLflow."""
    if not is_tracking_enabled(config):
        return None

    lm_eval_result_path = evaluation_result.get("lm_eval_result_path", None)
    if lm_eval_result_path is None:
        raise ValueError("evaluation_result.lm_eval_result_path is required.")

    setup_mlflow(config)
    parsed_lm_eval_result = parse_lm_eval_results(str(lm_eval_result_path))
    mlflow_run_id = evaluation_result.get("mlflow_run_id", None)

    if mlflow_run_id is None:
        run_name = str(evaluation_result.get("run_name", "evaluation"))
        with mlflow.start_run(experiment_id=MLFLOW_EXPERIMENT_ID, run_name=run_name) as active_run:
            log_evaluation_metrics(parsed_lm_eval_result)
            log_evaluation_tags(evaluation_result)
            log_evaluation_artifacts(evaluation_result)
            return str(active_run.info.run_id)

    try:
        with mlflow.start_run(run_id=str(mlflow_run_id)):
            log_evaluation_metrics(parsed_lm_eval_result)
            log_evaluation_tags(evaluation_result)
            log_evaluation_artifacts(evaluation_result)
            return str(mlflow_run_id)
    except MlflowException:
        run_name = str(evaluation_result.get("run_name", "evaluation"))
        with mlflow.start_run(experiment_id=MLFLOW_EXPERIMENT_ID, run_name=run_name) as active_run:
            mlflow.set_tag("evaluation.original_mlflow_run_id", str(mlflow_run_id))
            mlflow.set_tag("evaluation.mlflow_fallback", "missing_training_run")
            log_evaluation_metrics(parsed_lm_eval_result)
            log_evaluation_tags(evaluation_result)
            log_evaluation_artifacts(evaluation_result)
            return str(active_run.info.run_id)
