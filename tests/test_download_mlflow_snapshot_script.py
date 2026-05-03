import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


# /**
#  * Загружает download_mlflow_snapshot.py как тестируемый модуль.
#  *
#  * @return Python-модуль скрипта.
#  */
def load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "workflows" / "download_mlflow_snapshot.py"
    spec = importlib.util.spec_from_file_location("download_mlflow_snapshot", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load download_mlflow_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# /**
#  * Проверяет, что CLI скачивает snapshot и передает его в merge helper.
#  *
#  * @param monkeypatch Инструмент pytest для подмены функций.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_download_mlflow_snapshot_script_downloads_and_merges(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_script_module()
    archive_path = tmp_path / "downloaded" / "mlruns.zip"
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(module, "get_hf_token", lambda token_env_var: "token")
    monkeypatch.setattr(
        module,
        "download_file_from_hf",
        lambda **kwargs: calls.append({"download": kwargs}) or archive_path,
    )
    monkeypatch.setattr(
        module,
        "merge_mlflow_snapshot_archive",
        lambda **kwargs: calls.append({"merge": kwargs}) or tmp_path / "outputs" / "mlruns",
    )

    module.main(
        [
            "--repo-id",
            "user/repo",
            "--repo-path",
            "mlflow/mlruns.zip",
            "--mlruns-dir",
            str(tmp_path / "outputs" / "mlruns"),
            "--download-dir",
            str(tmp_path / "downloaded"),
            "--extract-dir",
            str(tmp_path / "extract"),
            "--token-env-var",
            "CUSTOM_TOKEN",
            "--revision",
            "abc",
        ]
    )

    assert calls == [
        {
            "download": {
                "repo_id": "user/repo",
                "target_dir": str(tmp_path / "downloaded"),
                "repo_path": "mlflow/mlruns.zip",
                "token": "token",
                "revision": "abc",
            }
        },
        {
            "merge": {
                "archive_path": archive_path,
                "target_mlruns_dir": str(tmp_path / "outputs" / "mlruns"),
                "extract_dir": str(tmp_path / "extract"),
            }
        },
    ]
