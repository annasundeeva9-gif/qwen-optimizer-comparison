"""Manual helper for logging already saved evaluation results to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from optimizer_comparison.artifacts.hf_hub import update_mlflow_experiment_artifact_locations
from optimizer_comparison.artifacts.local_store import save_json
from optimizer_comparison.tracking.mlflow_logger import log_evaluation_run


# /**
#  * Создает parser для повторного логирования уже готового evaluation-result в MLflow.
#  *
#  * @return Parser с аргументами run directory и путями evaluation artifacts.
#  */
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log existing evaluation result to MLflow.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--evaluation-result-path", default=None)
    parser.add_argument("--tracking-uri", default="outputs/mlruns")
    return parser


# /**
#  * Загружает JSON-файл как словарь.
#  *
#  * @param path Путь к JSON-файлу.
#  * @return Словарь из JSON-файла.
#  */
def load_json_dict(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


# /**
#  * Повторно логирует сохраненный evaluation-result в MLflow без запуска harness.
#  *
#  * @param argv CLI-аргументы или None для чтения из sys.argv.
#  * @return None.
#  */
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    config_path = Path(args.config_path) if args.config_path else run_dir / "config.yaml"
    evaluation_result_path = (
        Path(args.evaluation_result_path)
        if args.evaluation_result_path
        else run_dir / "evaluation" / "evaluation_result.json"
    )

    config = OmegaConf.load(config_path)
    config.tracking.enabled = True
    config.tracking.tracking_uri = args.tracking_uri
    update_mlflow_experiment_artifact_locations(args.tracking_uri)

    evaluation_result = load_json_dict(evaluation_result_path)
    mlflow_run_id = log_evaluation_run(config=config, evaluation_result=evaluation_result)
    if mlflow_run_id is not None:
        evaluation_result["mlflow_run_id"] = mlflow_run_id
        save_json(data=evaluation_result, output_path=evaluation_result_path)
        print(f"Logged evaluation to MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
