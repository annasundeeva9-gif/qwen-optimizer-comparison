import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from pytest import MonkeyPatch


# /**
#  * Загружает upload_mlflow_snapshot.py как тестируемый модуль.
#  *
#  * @return Python-модуль скрипта.
#  */
def load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "workflows" / "upload_mlflow_snapshot.py"
    spec = importlib.util.spec_from_file_location("upload_mlflow_snapshot", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load upload_mlflow_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# /**
#  * Проверяет CLI-аргументы upload_mlflow_snapshot.py.
#  *
#  * @return None.
#  */
def test_upload_mlflow_snapshot_parser_reads_arguments() -> None:
    upload_mlflow_snapshot = load_script_module()
    parser = upload_mlflow_snapshot.build_parser()

    args = parser.parse_args(
        [
            "--mlruns-dir",
            "outputs/mlruns",
            "--repo-id",
            "user/repo",
            "--repo-path",
            "mlflow/snapshot.zip",
            "--snapshot-dir",
            "outputs/snapshots",
            "--token-env-var",
            "CUSTOM_TOKEN",
        ]
    )

    assert args.mlruns_dir == "outputs/mlruns"
    assert args.repo_id == "user/repo"
    assert args.repo_path == "mlflow/snapshot.zip"
    assert args.snapshot_dir == "outputs/snapshots"
    assert args.token_env_var == "CUSTOM_TOKEN"


# /**
#  * Проверяет main upload_mlflow_snapshot.py без сетевого доступа.
#  *
#  * @param monkeypatch Инструмент pytest для подмены HF helpers.
#  * @return None.
#  */
def test_upload_mlflow_snapshot_main_calls_upload(
    monkeypatch: MonkeyPatch,
) -> None:
    upload_mlflow_snapshot = load_script_module()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(upload_mlflow_snapshot, "get_hf_token", lambda env_var: "token")
    monkeypatch.setattr(
        upload_mlflow_snapshot,
        "upload_mlflow_snapshot_to_hf",
        lambda **kwargs: calls.append(kwargs) or {"commit_url": "url", "revision": "rev"},
    )

    upload_mlflow_snapshot.main(
        [
            "--mlruns-dir",
            "outputs/mlruns",
            "--repo-id",
            "user/repo",
            "--repo-path",
            "mlflow/snapshot.zip",
            "--snapshot-dir",
            "outputs/snapshots",
        ]
    )

    assert calls == [
        {
            "mlruns_dir": "outputs/mlruns",
            "repo_id": "user/repo",
            "token": "token",
            "repo_path": "mlflow/snapshot.zip",
            "snapshot_dir": "outputs/snapshots",
        }
    ]
