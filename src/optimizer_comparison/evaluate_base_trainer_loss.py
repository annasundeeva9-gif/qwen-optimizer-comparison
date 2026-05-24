"""Entrypoint for evaluating base model validation loss with Trainer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import (
    resolve_project_path,
    save_json,
    save_resolved_config,
)
from optimizer_comparison.evaluation.model_source import (
    BASE_QWEN_MODEL_ID,
    BASE_QWEN_RUN_DIR,
    BASE_QWEN_RUN_ID,
    BASE_QWEN_RUN_NAME,
)
from optimizer_comparison.models.build_model import build_model
from optimizer_comparison.models.tokenization import build_tokenizer
from optimizer_comparison.tracking.mlflow_logger import log_run_config, setup_mlflow
from optimizer_comparison.training.seed import set_seed
from optimizer_comparison.training.training_loop import (
    build_standard_trainer,
    get_final_training_dataset,
)

TRAINER_EVAL_RESULT_FILENAME = "trainer_eval_result.json"
TRAINER_EVAL_CONFIG_FILENAME = "trainer_eval_config.yaml"


def load_existing_mlflow_run_id(result_path: Path) -> str | None:
    """Loads an existing MLflow run id from a previous trainer-eval result."""
    if not result_path.is_file():
        return None

    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None

    mlflow_run_id = data.get("mlflow_run_id", None)
    if mlflow_run_id is None:
        return None
    return str(mlflow_run_id)


def log_base_trainer_eval_to_mlflow(
    config: DictConfig,
    result: dict[str, Any],
    result_path: Path,
    config_path: Path,
    existing_mlflow_run_id: str | None,
) -> str | None:
    """Logs base model trainer-eval loss as an MLflow step-zero metric."""
    if not bool(config.get("tracking", {}).get("enabled", False)):
        return None

    setup_mlflow(config)
    metrics = result["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("Base trainer-eval result metrics must be a dictionary.")

    def write_active_run() -> str:
        log_run_config(config)
        mlflow.set_tag("run.status", str(result.get("status", "unknown")))
        mlflow.set_tag("project.run_id", str(result.get("run_id", "")))
        mlflow.set_tag("evaluation.kind", "trainer_eval_loss")
        mlflow.set_tag("evaluation.step", "0")
        mlflow.set_tag("model.base_model_id", BASE_QWEN_MODEL_ID)

        eval_loss = metrics.get("eval_loss", None)
        if isinstance(eval_loss, int | float):
            mlflow.log_metric("eval/loss", float(eval_loss), step=0)

        for metric_name, metric_value in metrics.items():
            if metric_name == "eval_loss":
                continue
            if isinstance(metric_value, int | float):
                mlflow.log_metric(f"eval/{metric_name}", float(metric_value), step=0)

        mlflow.log_artifact(str(result_path))
        mlflow.log_artifact(str(config_path))
        return str(mlflow.active_run().info.run_id)

    if existing_mlflow_run_id is not None:
        try:
            with mlflow.start_run(run_id=existing_mlflow_run_id):
                return write_active_run()
        except MlflowException:
            pass

    with mlflow.start_run(run_name=f"{BASE_QWEN_RUN_NAME}_trainer_eval_loss"):
        return write_active_run()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Runs Trainer.evaluate for the fixed base Qwen model on the project validation split."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for base model trainer-eval runs.")

    set_seed(int(config.training.seed))
    run_dir = resolve_project_path(BASE_QWEN_RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / TRAINER_EVAL_RESULT_FILENAME
    config_path = save_resolved_config(
        config=config,
        run_dir=run_dir,
        filename=TRAINER_EVAL_CONFIG_FILENAME,
    )

    start_time = time.perf_counter()
    dataset = get_final_training_dataset(config)
    tokenizer = build_tokenizer(config.model)
    model = build_model(config.model)
    trainer = build_standard_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        run_dir=run_dir,
    )
    metrics = {key: float(value) for key, value in trainer.evaluate().items()}
    elapsed_seconds = time.perf_counter() - start_time
    metrics["total_runtime_seconds"] = float(elapsed_seconds)

    existing_mlflow_run_id = load_existing_mlflow_run_id(result_path)
    result: dict[str, Any] = {
        "status": "completed",
        "run_id": BASE_QWEN_RUN_ID,
        "run_name": f"{BASE_QWEN_RUN_NAME}_trainer_eval_loss",
        "mlflow_run_id": None,
        "model_source": {
            "type": "base_model",
            "base_model_id": BASE_QWEN_MODEL_ID,
            "model_path": BASE_QWEN_MODEL_ID,
            "tokenizer_path": BASE_QWEN_MODEL_ID,
        },
        "metrics": metrics,
        "history": [{**metrics, "step": 0}],
        "artifacts": {
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "result_path": str(result_path),
        },
    }

    save_json(data=result, output_path=result_path)
    mlflow_run_id = log_base_trainer_eval_to_mlflow(
        config=config,
        result=result,
        result_path=result_path,
        config_path=config_path,
        existing_mlflow_run_id=existing_mlflow_run_id,
    )
    if mlflow_run_id is not None:
        result["mlflow_run_id"] = mlflow_run_id
        save_json(data=result, output_path=result_path)

    print({"result_path": str(result_path), "mlflow_run_id": result["mlflow_run_id"]})


if __name__ == "__main__":
    main()
