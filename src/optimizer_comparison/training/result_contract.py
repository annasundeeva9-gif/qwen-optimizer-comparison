"""Training result contract helpers."""

from __future__ import annotations

from pathlib import Path

TrainingResult = dict[str, object]


def build_empty_artifacts() -> dict[str, object]:
    """Creates the empty artifact structure for a training result."""
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
            "best_path": None,
            "last_path": None,
            "cleanup_status": "not_applicable",
            "cleanup_error": None,
            "removed_paths": [],
        },
        "plots": {
            "training_curves_path": None,
        },
        "debug": {
            "mezo_debug_steps_path": None,
            "mezo_debug_params_path": None,
        },
        "hf_hub": {
            "repo_id": None,
            "artifact_path": None,
            "revision": None,
            "commit_url": None,
            "result_commit_url": None,
            "upload_status": "not_applicable",
            "upload_error": None,
        },
    }


def build_training_result(
    run_name: str,
    status: str,
    final_loss: float | None,
    training_time_seconds: float | None,
    time_per_step_seconds: float | None,
    max_memory_mb: float | None,
) -> TrainingResult:
    """Creates a training result in the shared project format."""
    return {
        "run_id": None,
        "mlflow_run_id": None,
        "status": status,
        "run_name": run_name,
        "metrics": {
            "final_loss": final_loss,
            "training_time_seconds": training_time_seconds,
            "time_per_step_seconds": time_per_step_seconds,
            "max_memory_mb": max_memory_mb,
        },
        "history": [],
        "artifacts": build_empty_artifacts(),
    }


def set_local_artifact_paths(
    result: TrainingResult,
    run_dir: str | Path,
    config_path: str | Path,
    result_path: str | Path,
) -> TrainingResult:
    """Writes local run, config, and result paths into a training result."""
    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    result["run_id"] = Path(run_dir).name
    artifacts["run_dir"] = str(run_dir)
    artifacts["config_path"] = str(config_path)
    artifacts["result_path"] = str(result_path)
    return result
