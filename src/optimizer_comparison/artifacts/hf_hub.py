"""Hugging Face Hub artifact persistence helpers."""

from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import save_json
from optimizer_comparison.training.result_contract import TrainingResult


def should_use_hf_hub(config: DictConfig) -> bool:
    """Checks whether Hugging Face Hub artifact persistence is enabled."""
    return bool(config.get("artifacts", {}).get("hf_hub", {}).get("use", False))


def get_hf_token(token_env_var: str) -> str:
    """Reads the Hugging Face token from an environment variable."""
    token = os.environ.get(token_env_var)
    if not token:
        raise ValueError(f"Hugging Face token env var is not set: {token_env_var}")
    return token


def validate_hf_hub_before_training(config: DictConfig) -> None:
    """Validates Hugging Face Hub settings before real training starts."""
    hf_hub_config = config.get("artifacts", {}).get("hf_hub", {})
    use_hf_hub = bool(hf_hub_config.get("use", False))

    if not use_hf_hub:
        return

    repo_id = hf_hub_config.get("repo_id", None)
    if repo_id is None:
        raise ValueError("artifacts.hf_hub.repo_id is not set.")

    get_hf_token(str(hf_hub_config.get("token_env_var", "HF_TOKEN")))


def upload_path_to_hf(
    artifact_path: str | Path,
    repo_id: str,
    repo_path: str,
    token: str,
    commit_message: str = "Upload training artifact",
    api: HfApi | None = None,
) -> dict[str, str | None]:
    """Uploads a local file or directory to Hugging Face Hub."""
    hf_api = api or HfApi()
    local_path = Path(artifact_path)

    if local_path.is_dir():
        commit_info = hf_api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            path_in_repo=repo_path,
            token=token,
            commit_message=commit_message,
        )
    elif local_path.is_file():
        commit_info = hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            token=token,
            commit_message=commit_message,
        )
    else:
        raise FileNotFoundError(f"Artifact path does not exist: {artifact_path}")

    return {
        "commit_url": str(getattr(commit_info, "commit_url", "") or ""),
        "revision": str(getattr(commit_info, "oid", "") or ""),
    }


def download_artifacts_from_hf(
    repo_id: str,
    target_dir: str | Path,
    repo_path: str,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """Downloads run artifacts from Hugging Face Hub into a local directory."""
    target_path = Path(target_dir)
    allow_patterns = [f"{repo_path}/**"] if repo_path else None
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(target_path),
        allow_patterns=allow_patterns,
    )
    return target_path / repo_path if repo_path else target_path


def download_file_from_hf(
    repo_id: str,
    target_dir: str | Path,
    repo_path: str,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """Downloads one file from Hugging Face Hub into a local directory."""
    target_path = Path(target_dir)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(target_path),
        allow_patterns=[repo_path],
    )
    downloaded_path = target_path / repo_path
    if not downloaded_path.is_file():
        raise FileNotFoundError(
            "HF file artifact was not downloaded. "
            f"Expected local file: {downloaded_path}. "
            f"Check that repo_path exists in the HF repo: {repo_path}"
        )
    return downloaded_path


def merge_directory_contents(source_dir: str | Path, target_dir: str | Path) -> None:
    """Merges one directory into another without deleting existing subdirectories."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory for merge does not exist: {source_dir}")

    target_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)


def update_mlflow_experiment_artifact_locations(mlruns_dir: str | Path) -> None:
    """Updates MLflow experiment artifact locations after moving a file store."""
    root = Path(mlruns_dir)
    if not root.is_dir():
        return

    for experiment_dir in root.iterdir():
        if not experiment_dir.is_dir() or not experiment_dir.name.isdigit():
            continue

        meta_path = experiment_dir / "meta.yaml"
        if not meta_path.is_file():
            continue

        lines = meta_path.read_text(encoding="utf-8").splitlines()
        artifact_uri = experiment_dir.resolve().as_uri()
        updated_lines = [
            f"artifact_location: {artifact_uri}"
            if line.startswith("artifact_location:")
            else line
            for line in lines
        ]
        meta_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def merge_mlflow_snapshot_archive(
    archive_path: str | Path,
    target_mlruns_dir: str | Path,
    extract_dir: str | Path,
) -> Path:
    """Extracts a zipped MLflow snapshot and merges it with the local file store."""
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"MLflow snapshot archive does not exist: {archive_path}")

    extract_path = Path(extract_dir)
    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(extract_path)

    extracted_mlruns = extract_path / "mlruns"
    if not extracted_mlruns.is_dir():
        raise FileNotFoundError(
            f"MLflow snapshot must contain top-level mlruns directory: {archive_path}"
        )

    target_path = Path(target_mlruns_dir)
    merge_directory_contents(source_dir=extracted_mlruns, target_dir=target_path)
    update_mlflow_experiment_artifact_locations(target_path)
    return target_path


def create_mlflow_snapshot_archive(
    mlruns_dir: str | Path,
    output_dir: str | Path,
    archive_name: str = "mlruns_snapshot",
) -> Path:
    """Creates a zip archive from the local MLflow file store."""
    mlruns_path = Path(mlruns_dir)
    if not mlruns_path.is_dir():
        raise FileNotFoundError(f"MLflow directory does not exist: {mlruns_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    archive_path = output_path / f"{archive_name}.zip"
    if archive_path.exists():
        archive_path.unlink()

    created_archive = shutil.make_archive(
        base_name=str(output_path / archive_name),
        format="zip",
        root_dir=str(mlruns_path.parent),
        base_dir=mlruns_path.name,
    )
    return Path(created_archive)


def upload_mlflow_snapshot_to_hf(
    mlruns_dir: str | Path,
    repo_id: str,
    token: str,
    repo_path: str = "mlflow/mlruns_snapshot.zip",
    snapshot_dir: str | Path = "outputs/mlflow_snapshots",
) -> dict[str, str | None]:
    """Uploads a zipped MLflow file store snapshot to Hugging Face Hub."""
    archive_path = create_mlflow_snapshot_archive(
        mlruns_dir=mlruns_dir,
        output_dir=snapshot_dir,
    )
    return upload_path_to_hf(
        artifact_path=archive_path,
        repo_id=repo_id,
        repo_path=repo_path,
        token=token,
        commit_message="Upload MLflow snapshot",
    )


def set_hf_upload_status(
    result: TrainingResult,
    status: str,
    error: str | None = None,
) -> TrainingResult:
    """Writes Hugging Face Hub upload status into the training result."""
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    hf_hub = artifacts.get("hf_hub", {})
    if not isinstance(hf_hub, dict):
        raise TypeError("Training result hf_hub artifacts must be a dictionary.")

    hf_hub["upload_status"] = status
    hf_hub["upload_error"] = error

    model_artifacts = artifacts.get("model", {})
    if isinstance(model_artifacts, dict):
        model_artifacts["upload_status"] = status
        model_artifacts["upload_error"] = error

    return result


def cleanup_local_checkpoint_artifacts(result: TrainingResult) -> TrainingResult:
    """Removes local checkpoint artifacts after a successful Hugging Face Hub upload."""
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    checkpoint_artifacts = artifacts.get("checkpoints", {})
    if not isinstance(checkpoint_artifacts, dict):
        raise TypeError("Training result checkpoint artifacts must be a dictionary.")

    run_dir = artifacts.get("run_dir", None)
    paths_to_remove: list[Path] = []

    checkpoint_root = checkpoint_artifacts.get("local_path", None)
    if checkpoint_root is not None:
        paths_to_remove.append(Path(str(checkpoint_root)))

    if run_dir is not None:
        paths_to_remove.append(Path(str(run_dir)) / "trainer_output")

    removed_paths: list[str] = []
    try:
        for path in paths_to_remove:
            if path.exists():
                shutil.rmtree(path)
                removed_paths.append(str(path))
    except Exception as error:
        checkpoint_artifacts["cleanup_status"] = "failed"
        checkpoint_artifacts["cleanup_error"] = str(error)
        checkpoint_artifacts["removed_paths"] = removed_paths
        return result

    checkpoint_artifacts["cleanup_status"] = "completed"
    checkpoint_artifacts["cleanup_error"] = None
    checkpoint_artifacts["removed_paths"] = removed_paths
    checkpoint_artifacts["local_path"] = None
    checkpoint_artifacts["best_path"] = None
    checkpoint_artifacts["last_path"] = None
    return result


def persist_training_artifacts_to_hf(
    config: DictConfig,
    result: TrainingResult,
) -> TrainingResult:
    """Uploads the main training run artifacts to Hugging Face Hub."""
    if not should_use_hf_hub(config):
        return set_hf_upload_status(result=result, status="skipped")

    artifacts_config = config.artifacts.hf_hub
    repo_id = artifacts_config.get("repo_id", None)
    if repo_id is None:
        error = "artifacts.hf_hub.repo_id is not set."
        set_hf_upload_status(result=result, status="failed", error=error)
        raise ValueError(error)

    try:
        token = get_hf_token(str(artifacts_config.token_env_var))
        metadata = upload_training_artifact_paths(
            result=result,
            repo_id=str(repo_id),
            token=token,
            experiment_name=str(config.experiment.name),
        )
    except Exception as error:
        set_hf_upload_status(result=result, status="failed", error=str(error))
        warnings.warn(f"HF Hub upload failed: {error}", RuntimeWarning, stacklevel=2)
        return result

    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    hf_hub = artifacts["hf_hub"]
    if not isinstance(hf_hub, dict):
        raise TypeError("Training result hf_hub artifacts must be a dictionary.")

    hf_hub.update(metadata)
    hf_hub["upload_status"] = "completed"
    hf_hub["upload_error"] = None

    model_artifacts = artifacts["model"]
    if isinstance(model_artifacts, dict):
        model_artifacts["hf_repo_id"] = str(repo_id)
        model_artifacts["hf_commit_url"] = metadata["commit_url"]
        model_artifacts["upload_status"] = "completed"
        model_artifacts["upload_error"] = None

    result_path = artifacts.get("result_path")
    if result_path is not None:
        try:
            save_json(data=result, output_path=str(result_path))
            result_metadata = upload_path_to_hf(
                artifact_path=str(result_path),
                repo_id=str(repo_id),
                repo_path=f"{metadata['artifact_path']}/result.json",
                token=token,
                commit_message="Upload final training result metadata",
            )
            hf_hub["result_commit_url"] = result_metadata["commit_url"]
        except Exception as error:
            set_hf_upload_status(result=result, status="failed", error=str(error))
            warnings.warn(f"HF Hub upload failed: {error}", RuntimeWarning, stacklevel=2)
            return result

    if bool(config.artifacts.get("cleanup_checkpoints_after_hf_upload", True)):
        cleanup_local_checkpoint_artifacts(result)

    return result


def upload_training_artifact_paths(
    result: TrainingResult,
    repo_id: str,
    token: str,
    experiment_name: str,
) -> dict[str, str | None]:
    """Uploads local model, tokenizer, config, and result artifacts for one run."""
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    run_dir = artifacts.get("run_dir", None)
    if run_dir is None:
        raise ValueError("Training result artifacts.run_dir must be set before HF upload.")

    run_id = str(result.get("run_id") or Path(str(run_dir)).name)
    artifact_path = f"runs/{run_id}"
    model_artifacts = artifacts.get("model", {})
    tokenizer_artifacts = artifacts.get("tokenizer", {})
    uploads = {
        "model": (
            model_artifacts.get("local_path") if isinstance(model_artifacts, dict) else None
        ),
        "tokenizer": (
            tokenizer_artifacts.get("local_path")
            if isinstance(tokenizer_artifacts, dict)
            else None
        ),
        "config.yaml": artifacts.get("config_path"),
        "result.json": artifacts.get("result_path"),
    }

    last_metadata: dict[str, str | None] = {"commit_url": None, "revision": None}
    for repo_name, local_path in uploads.items():
        if local_path is None:
            continue
        last_metadata = upload_path_to_hf(
            artifact_path=str(local_path),
            repo_id=repo_id,
            repo_path=f"{artifact_path}/{repo_name}",
            token=token,
            commit_message=f"Upload {repo_name} for {experiment_name}",
        )

    return {
        "repo_id": repo_id,
        "artifact_path": artifact_path,
        "commit_url": last_metadata["commit_url"],
        "revision": last_metadata["revision"],
        "result_commit_url": None,
    }
