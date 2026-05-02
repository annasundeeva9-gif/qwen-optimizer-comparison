"""Entrypoint for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import save_json
from optimizer_comparison.evaluation.harness_runner import run_lm_eval_harness
from optimizer_comparison.evaluation.model_source import (
    EvaluationModelSource,
    resolve_evaluation_model_source,
)
from optimizer_comparison.tracking.mlflow_logger import log_evaluation_run


# /**
#  * Загружает training result из run directory, к которому относится model source.
#  *
#  * @param model_source Локально доступные model/tokenizer artifacts.
#  * @return Training result из result.json.
#  */
def load_training_result_for_evaluation(
    model_source: EvaluationModelSource,
) -> dict[str, Any]:
    if model_source.run_dir is None:
        raise ValueError("Evaluation model source must provide run_dir.")

    result_path = model_source.run_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Training result.json does not exist: {result_path}")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Training result.json must contain a JSON object.")
    return data


# /**
#  * Создает краткий result evaluation-запуска для локального сохранения и MLflow logging.
#  *
#  * @param training_result Training result исходной модели.
#  * @param model_source Локально доступные model/tokenizer artifacts.
#  * @param harness_paths Пути, созданные runner-ом lm-evaluation-harness.
#  * @return Evaluation result в проектном формате.
#  */
def build_evaluation_result(
    training_result: dict[str, Any],
    model_source: EvaluationModelSource,
    harness_paths: dict[str, str],
) -> dict[str, Any]:
    run_id = training_result.get("run_id", None)
    mlflow_run_id = training_result.get("mlflow_run_id", None)
    if run_id is None:
        raise ValueError("Training result must contain run_id before evaluation.")
    if mlflow_run_id is None:
        raise ValueError("Training result must contain mlflow_run_id before evaluation.")

    return {
        "status": "completed",
        "run_id": str(run_id),
        "run_name": str(training_result.get("run_name", run_id)),
        "mlflow_run_id": str(mlflow_run_id),
        "model_source": {
            "type": model_source.source_type,
            "run_dir": None if model_source.run_dir is None else str(model_source.run_dir),
            "model_path": str(model_source.model_path),
            "tokenizer_path": str(model_source.tokenizer_path),
            "hf_repo_id": model_source.hf_repo_id,
            "hf_artifact_path": model_source.hf_artifact_path,
            "hf_revision": model_source.hf_revision,
        },
        "lm_eval_result_path": harness_paths["output_path"],
        "raw_log_path": harness_paths["raw_log_path"],
    }


# /**
#  * Возвращает путь для project-level evaluation result.
#  *
#  * @param config Полная конфигурация запуска.
#  * @param model_source Локально доступные model/tokenizer artifacts.
#  * @return Путь к evaluation_result.json.
#  */
def get_evaluation_result_path(config: DictConfig, model_source: EvaluationModelSource) -> Path:
    if model_source.run_dir is None:
        raise ValueError("Evaluation model source must provide run_dir.")

    result_filename = str(config.evaluation.harness.result_filename)
    return model_source.run_dir / "evaluation" / result_filename


# /**
#  * Запускает evaluation-пайплайн поверх lm-evaluation-harness.
#  *
#  * @param config Полная конфигурация запуска, собранная Hydra.
#  * @return None. Результаты сохраняются в evaluation-директории training run-а.
#  */
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    model_source = resolve_evaluation_model_source(config)
    training_result = load_training_result_for_evaluation(model_source)
    harness_paths = run_lm_eval_harness(config=config, model_source=model_source)
    evaluation_result = build_evaluation_result(
        training_result=training_result,
        model_source=model_source,
        harness_paths=harness_paths,
    )
    evaluation_result_path = get_evaluation_result_path(config=config, model_source=model_source)
    save_json(data=evaluation_result, output_path=evaluation_result_path)
    log_evaluation_run(config=config, evaluation_result=evaluation_result)


if __name__ == "__main__":
    main()
