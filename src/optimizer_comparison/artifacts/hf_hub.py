"""Hugging Face Hub artifact persistence helpers."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import save_json
from optimizer_comparison.training.result_contract import TrainingResult


# /**
#  * Возвращает true, если удаленное сохранение включено в конфиге.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @return True, если artifacts.hf_hub.use=true.
#  */
def should_use_hf_hub(config: DictConfig) -> bool:
    return bool(config.get("artifacts", {}).get("hf_hub", {}).get("use", False))


# /**
#  * Возвращает HF token из переменной окружения.
#  *
#  * @param token_env_var Имя переменной окружения с HF token.
#  * @return Значение token.
#  */
def get_hf_token(token_env_var: str) -> str:
    token = os.environ.get(token_env_var)
    if not token:
        raise ValueError(f"Hugging Face token env var is not set: {token_env_var}")
    return token


# /**
#  * Проверяет HF Hub настройки до начала реального обучения.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @return None.
#  */
def validate_hf_hub_before_training(config: DictConfig) -> None:
    hf_hub_config = config.get("artifacts", {}).get("hf_hub", {})
    use_hf_hub = bool(hf_hub_config.get("use", False))

    if not use_hf_hub:
        return

    repo_id = hf_hub_config.get("repo_id", None)
    if repo_id is None:
        raise ValueError("artifacts.hf_hub.repo_id is not set.")

    get_hf_token(str(hf_hub_config.get("token_env_var", "HF_TOKEN")))


# /**
#  * Загружает файл или директорию в Hugging Face Hub.
#  *
#  * @param artifact_path Локальный файл или директория.
#  * @param repo_id Идентификатор HF repo.
#  * @param repo_path Путь внутри repo.
#  * @param token HF token.
#  * @param commit_message Сообщение commit-а.
#  * @param api Опциональный HfApi для тестов.
#  * @return Метаданные commit-а.
#  */
def upload_path_to_hf(
    artifact_path: str | Path,
    repo_id: str,
    repo_path: str,
    token: str,
    commit_message: str = "Upload training artifact",
    api: HfApi | None = None,
) -> dict[str, str | None]:
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


# /**
#  * Загружает run artifacts из Hugging Face Hub в локальную директорию.
#  *
#  * @param repo_id Идентификатор HF repo.
#  * @param target_dir Локальная директория для восстановления artifacts.
#  * @param repo_path Путь artifacts внутри repo.
#  * @param token HF token или None для публичного repo.
#  * @param revision Revision или commit sha.
#  * @return Путь к восстановленной локальной директории.
#  */
def download_artifacts_from_hf(
    repo_id: str,
    target_dir: str | Path,
    repo_path: str,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
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


# /**
#  * Создает zip-архив локального MLflow file store.
#  *
#  * @param mlruns_dir Локальная директория MLflow file store.
#  * @param output_dir Директория для snapshot-архива.
#  * @param archive_name Имя архива без расширения.
#  * @return Путь к созданному zip-архиву.
#  */
def create_mlflow_snapshot_archive(
    mlruns_dir: str | Path,
    output_dir: str | Path,
    archive_name: str = "mlruns_snapshot",
) -> Path:
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


# /**
#  * Загружает zip snapshot MLflow file store в Hugging Face Hub.
#  *
#  * @param mlruns_dir Локальная директория MLflow file store.
#  * @param repo_id Идентификатор HF repo.
#  * @param token HF token.
#  * @param repo_path Путь zip-архива внутри repo.
#  * @param snapshot_dir Локальная директория для временного snapshot-архива.
#  * @return Метаданные upload-а.
#  */
def upload_mlflow_snapshot_to_hf(
    mlruns_dir: str | Path,
    repo_id: str,
    token: str,
    repo_path: str = "mlflow/mlruns_snapshot.zip",
    snapshot_dir: str | Path = "outputs/mlflow_snapshots",
) -> dict[str, str | None]:
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


# /**
#  * Записывает статус HF Hub upload в training-result.
#  *
#  * @param result Training-result, который нужно обновить.
#  * @param status Статус upload.
#  * @param error Текст ошибки или None.
#  * @return Обновленный training-result.
#  */
def set_hf_upload_status(
    result: TrainingResult,
    status: str,
    error: str | None = None,
) -> TrainingResult:
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


# /**
#  * Удаляет локальные resume-checkpoint artifacts после успешного HF Hub upload.
#  *
#  * @param result Training-result с локальными путями artifacts.
#  * @return Training-result с обновленным статусом cleanup-а.
#  */
def cleanup_local_checkpoint_artifacts(result: TrainingResult) -> TrainingResult:
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


# /**
#  * Загружает основные artifacts training run-а в Hugging Face Hub.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @param result Training-result с локальными путями artifacts.
#  * @return Training-result с HF Hub metadata.
#  */
def persist_training_artifacts_to_hf(
    config: DictConfig,
    result: TrainingResult,
) -> TrainingResult:
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

    cleanup_local_checkpoint_artifacts(result)

    return result


# /**
#  * Загружает локальные model/tokenizer/config/result artifacts одного run-а.
#  *
#  * @param result Training-result с локальными путями artifacts.
#  * @param repo_id Идентификатор HF repo.
#  * @param token HF token.
#  * @param experiment_name Имя эксперимента.
#  * @return HF Hub metadata для result.json и MLflow.
#  */
def upload_training_artifact_paths(
    result: TrainingResult,
    repo_id: str,
    token: str,
    experiment_name: str,
) -> dict[str, str | None]:
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
