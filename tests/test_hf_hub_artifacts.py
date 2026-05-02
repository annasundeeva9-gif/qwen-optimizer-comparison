from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from optimizer_comparison.artifacts import hf_hub
from optimizer_comparison.artifacts.hf_hub import (
    cleanup_local_checkpoint_artifacts,
    create_mlflow_snapshot_archive,
    download_artifacts_from_hf,
    persist_training_artifacts_to_hf,
    upload_mlflow_snapshot_to_hf,
    upload_path_to_hf,
    validate_hf_hub_before_training,
)
from optimizer_comparison.training.result_contract import build_training_result


# /**
#  * Fake HfApi для проверки upload без сети.
#  */
class FakeHfApi:
    def __init__(self) -> None:
        self.created_repos: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []

    # /**
    #  * Имитирует создание HF repo.
    #  *
    #  * @param kwargs Аргументы create_repo.
    #  * @return None.
    #  */
    def create_repo(self, **kwargs: Any) -> None:
        self.created_repos.append(kwargs)

    # /**
    #  * Имитирует upload директории.
    #  *
    #  * @param kwargs Аргументы upload_folder.
    #  * @return Fake commit info.
    #  */
    def upload_folder(self, **kwargs: Any) -> SimpleNamespace:
        self.uploads.append({"kind": "folder", **kwargs})
        return SimpleNamespace(
            commit_url="https://huggingface.co/user/repo/commit/folder",
            oid="abc",
        )

    # /**
    #  * Имитирует upload файла.
    #  *
    #  * @param kwargs Аргументы upload_file.
    #  * @return Fake commit info.
    #  */
    def upload_file(self, **kwargs: Any) -> SimpleNamespace:
        self.uploads.append({"kind": "file", **kwargs})
        return SimpleNamespace(commit_url="https://huggingface.co/user/repo/commit/file", oid="def")


# /**
#  * Создает минимальный Hydra-like config с HF Hub секцией.
#  *
#  * @param use Флаг artifacts.hf_hub.use.
#  * @return Полный config.
#  */
def make_config(use: bool) -> Any:
    return OmegaConf.create(
        {
            "mode": {"name": "full"},
            "experiment": {"name": "adamw_baseline"},
            "artifacts": {
                "hf_hub": {
                    "use": use,
                    "repo_id": "user/repo",
                    "token_env_var": "HF_TOKEN",
                }
            },
        }
    )


# /**
#  * Проверяет, что smoke training можно запустить без HF Hub persistence.
#  *
#  * @return None.
#  */
def test_validate_hf_hub_before_training_allows_local_smoke() -> None:
    config = make_config(use=False)
    config.mode.name = "smoke"

    validate_hf_hub_before_training(config)


# /**
#  * Проверяет, что включенный HF Hub требует repo id до старта обучения.
#  *
#  * @param monkeypatch Инструмент pytest для окружения.
#  * @return None.
#  */
def test_validate_hf_hub_before_training_requires_repo_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")
    config = make_config(use=True)
    config.artifacts.hf_hub.repo_id = None

    with pytest.raises(ValueError, match="repo_id"):
        validate_hf_hub_before_training(config)


# /**
#  * Проверяет, что включенный HF Hub требует token до старта обучения.
#  *
#  * @param monkeypatch Инструмент pytest для окружения.
#  * @return None.
#  */
def test_validate_hf_hub_before_training_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = make_config(use=True)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        validate_hf_hub_before_training(config)


# /**
#  * Создает training-result с локальными путями artifacts.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return Training-result.
#  */
def make_result(tmp_path: Path) -> dict[str, object]:
    result = build_training_result(
        run_name="adamw_baseline",
        status="completed",
        final_loss=1.0,
        training_time_seconds=10.0,
        max_memory_mb=100.0,
    )
    run_id = "adamw_baseline__2026-04-29_10-00-00"
    result["run_id"] = run_id
    run_dir = tmp_path / run_id
    model_dir = run_dir / "model"
    tokenizer_dir = run_dir / "tokenizer"
    checkpoints_dir = run_dir / "checkpoints"
    trainer_checkpoint_dir = run_dir / "trainer_output" / "checkpoint-1"
    model_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (checkpoints_dir / "best").mkdir(parents=True)
    (checkpoints_dir / "last").mkdir()
    trainer_checkpoint_dir.mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    result_path = run_dir / "result.json"
    config_path.write_text("config: true\n", encoding="utf-8")
    result_path.write_text("{}", encoding="utf-8")

    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["run_dir"] = str(run_dir)
    artifacts["config_path"] = str(config_path)
    artifacts["result_path"] = str(result_path)
    assert isinstance(artifacts["model"], dict)
    assert isinstance(artifacts["tokenizer"], dict)
    assert isinstance(artifacts["checkpoints"], dict)
    artifacts["model"]["local_path"] = str(model_dir)
    artifacts["tokenizer"]["local_path"] = str(tokenizer_dir)
    artifacts["checkpoints"]["local_path"] = str(checkpoints_dir)
    artifacts["checkpoints"]["best_path"] = str(checkpoints_dir / "best")
    artifacts["checkpoints"]["last_path"] = str(checkpoints_dir / "last")
    return result


# /**
#  * Проверяет upload файла и директории через HfApi.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_upload_path_to_hf_supports_file_and_directory(tmp_path: Path) -> None:
    api = FakeHfApi()
    file_path = tmp_path / "result.json"
    dir_path = tmp_path / "model"
    file_path.write_text("{}", encoding="utf-8")
    dir_path.mkdir()

    file_metadata = upload_path_to_hf(
        artifact_path=file_path,
        repo_id="user/repo",
        repo_path="runs/run/result.json",
        token="token",
        api=api,
    )
    dir_metadata = upload_path_to_hf(
        artifact_path=dir_path,
        repo_id="user/repo",
        repo_path="runs/run/model",
        token="token",
        api=api,
    )

    assert file_metadata["revision"] == "def"
    assert dir_metadata["revision"] == "abc"
    assert api.created_repos == []
    assert api.uploads[0]["kind"] == "file"
    assert api.uploads[1]["kind"] == "folder"


# /**
#  * Проверяет, что при use=false upload пропускается.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_persist_training_artifacts_skips_when_hf_hub_is_not_used(tmp_path: Path) -> None:
    result = persist_training_artifacts_to_hf(
        config=make_config(use=False),
        result=make_result(tmp_path),
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["hf_hub"]["upload_status"] == "skipped"


# /**
#  * Проверяет, что use=true требует token.
#  *
#  * @param monkeypatch Инструмент pytest для окружения.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_persist_training_artifacts_requires_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = make_result(tmp_path)

    with pytest.warns(RuntimeWarning, match="HF_TOKEN"):
        result = persist_training_artifacts_to_hf(config=make_config(use=True), result=result)

    assert result["artifacts"]["hf_hub"]["upload_status"] == "failed"
    artifacts = result["artifacts"]
    assert Path(artifacts["checkpoints"]["local_path"]).is_dir()
    assert (Path(artifacts["run_dir"]) / "trainer_output" / "checkpoint-1").is_dir()


# /**
#  * Проверяет успешное заполнение HF Hub metadata без сети.
#  *
#  * @param monkeypatch Инструмент pytest для подмены upload.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_persist_training_artifacts_writes_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(
        hf_hub,
        "upload_path_to_hf",
        lambda **kwargs: {
            "commit_url": "https://huggingface.co/user/repo/commit/abc",
            "revision": "abc",
        },
    )

    result = persist_training_artifacts_to_hf(
        config=make_config(use=True),
        result=make_result(tmp_path),
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["hf_hub"]["upload_status"] == "completed"
    assert artifacts["hf_hub"]["repo_id"] == "user/repo"
    assert artifacts["hf_hub"]["artifact_path"] == "runs/adamw_baseline__2026-04-29_10-00-00"
    assert artifacts["model"]["hf_repo_id"] == "user/repo"
    assert artifacts["checkpoints"]["cleanup_status"] == "completed"
    assert artifacts["checkpoints"]["local_path"] is None
    assert not (tmp_path / "adamw_baseline__2026-04-29_10-00-00" / "checkpoints").exists()
    assert not (tmp_path / "adamw_baseline__2026-04-29_10-00-00" / "trainer_output").exists()


# /**
#  * Проверяет локальный cleanup checkpoint artifacts без HF Hub вызовов.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_cleanup_local_checkpoint_artifacts_removes_checkpoint_directories(
    tmp_path: Path,
) -> None:
    result = make_result(tmp_path)
    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)

    cleaned_result = cleanup_local_checkpoint_artifacts(result)

    checkpoint_artifacts = cleaned_result["artifacts"]["checkpoints"]
    assert checkpoint_artifacts["cleanup_status"] == "completed"
    assert checkpoint_artifacts["cleanup_error"] is None
    assert checkpoint_artifacts["local_path"] is None
    assert checkpoint_artifacts["best_path"] is None
    assert checkpoint_artifacts["last_path"] is None
    assert len(checkpoint_artifacts["removed_paths"]) == 2


# /**
#  * Проверяет download helper без сетевого доступа.
#  *
#  * @param monkeypatch Инструмент pytest для подмены snapshot_download.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_download_artifacts_from_hf_uses_repo_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(hf_hub, "snapshot_download", fake_snapshot_download)

    restored_path = download_artifacts_from_hf(
        repo_id="user/repo",
        target_dir=tmp_path,
        repo_path="runs/adamw/run",
        token="token",
        revision="abc",
    )

    assert restored_path == tmp_path / "runs/adamw/run"
    assert calls[0]["allow_patterns"] == ["runs/adamw/run/**"]
    assert calls[0]["revision"] == "abc"


# /**
#  * Проверяет создание zip snapshot-а MLflow file store.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_create_mlflow_snapshot_archive_creates_zip(tmp_path: Path) -> None:
    mlruns_dir = tmp_path / "mlruns"
    run_metric = mlruns_dir / "1" / "run" / "metrics" / "loss"
    run_metric.parent.mkdir(parents=True)
    run_metric.write_text("1 0.5 1\n", encoding="utf-8")

    archive_path = create_mlflow_snapshot_archive(
        mlruns_dir=mlruns_dir,
        output_dir=tmp_path / "snapshots",
    )

    assert archive_path == tmp_path / "snapshots" / "mlruns_snapshot.zip"
    assert archive_path.is_file()


# /**
#  * Проверяет upload MLflow snapshot-а через общий HF upload helper.
#  *
#  * @param monkeypatch Инструмент pytest для подмены upload.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_upload_mlflow_snapshot_to_hf_uploads_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_upload_path_to_hf(**kwargs: Any) -> dict[str, str | None]:
        calls.append(kwargs)
        return {"commit_url": "url", "revision": "rev"}

    monkeypatch.setattr(hf_hub, "upload_path_to_hf", fake_upload_path_to_hf)

    metadata = upload_mlflow_snapshot_to_hf(
        mlruns_dir=mlruns_dir,
        repo_id="user/repo",
        token="token",
        repo_path="mlflow/mlruns_after_training.zip",
        snapshot_dir=tmp_path / "snapshots",
    )

    assert metadata == {"commit_url": "url", "revision": "rev"}
    assert calls[0]["repo_id"] == "user/repo"
    assert calls[0]["repo_path"] == "mlflow/mlruns_after_training.zip"
    assert Path(calls[0]["artifact_path"]).is_file()
