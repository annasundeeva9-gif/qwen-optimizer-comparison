"""Manual helper for logging already saved training results to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from omegaconf import OmegaConf

from optimizer_comparison.tracking.mlflow_logger import (
    log_hf_hub_tags,
    log_run_config,
    log_training_history,
    log_training_metrics,
    setup_mlflow,
)
from optimizer_comparison.training.plots import save_training_curves_plot


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI parser for logging an existing training result."""
    parser = argparse.ArgumentParser(description="Log existing training result to MLflow.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--tracking-uri", default="outputs/mlruns")
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument("--extra-artifact", action="append", default=[])
    return parser


def load_json_dict(path: Path) -> dict[str, object]:
    """Loads a JSON file as a dictionary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


def main(argv: list[str] | None = None) -> None:
    """Re-logs a saved training result in MLflow without running training."""
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    config_path = Path(args.config_path) if args.config_path else run_dir / "config.yaml"
    result_path = Path(args.result_path) if args.result_path else run_dir / "result.json"

    config = OmegaConf.load(config_path)
    config.tracking.enabled = True
    config.tracking.tracking_uri = args.tracking_uri
    setup_mlflow(config)

    training_result = load_json_dict(result_path)
    mlflow_run_id = args.mlflow_run_id or training_result.get("mlflow_run_id", None)
    if mlflow_run_id is None:
        raise ValueError("--mlflow-run-id is required when result.json has no mlflow_run_id.")

    history = training_result.get("history", [])
    if not isinstance(history, list):
        raise TypeError("Training result history must be a list.")
    training_curves_path = save_training_curves_plot(
        history=history,
        output_path=run_dir / "training_curves.png",
    )

    with mlflow.start_run(run_id=str(mlflow_run_id)):
        log_run_config(config)
        mlflow.set_tag("run.status", str(training_result.get("status", "unknown")))
        mlflow.set_tag("project.run_id", str(training_result.get("run_id", "")))
        mlflow.set_tag("training.backfilled_from_result_json", str(result_path))
        log_training_metrics(training_result)
        log_training_history(training_result)
        log_hf_hub_tags(training_result)

        artifact_paths = [training_curves_path, *[Path(path) for path in args.extra_artifact]]
        for artifact_path in artifact_paths:
            if artifact_path is not None and artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path))

    print(f"Logged training to MLflow run: {mlflow_run_id}")


if __name__ == "__main__":
    main()
