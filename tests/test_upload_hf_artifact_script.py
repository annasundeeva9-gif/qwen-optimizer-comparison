import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


# /**
#  * Загружает upload_hf_artifact.py как тестируемый модуль.
#  *
#  * @return Python-модуль скрипта.
#  */
def load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "upload_hf_artifact.py"
    spec = importlib.util.spec_from_file_location("upload_hf_artifact", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load upload_hf_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# /**
#  * Проверяет, что ручной upload-скрипт передает CLI-аргументы в HF helper.
#  *
#  * @param monkeypatch Инструмент pytest для подмены функций.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_upload_hf_artifact_script_calls_upload_helper(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_script_module()
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(module, "get_hf_token", lambda token_env_var: "token")
    monkeypatch.setattr(
        module,
        "upload_path_to_hf",
        lambda **kwargs: calls.append(kwargs) or {"commit_url": "url", "revision": "rev"},
    )

    module.main(
        [
            "--artifact-path",
            str(artifact_path),
            "--repo-id",
            "user/repo",
            "--repo-path",
            "runs/manual/artifact",
            "--token-env-var",
            "CUSTOM_TOKEN",
        ]
    )

    assert calls == [
        {
            "artifact_path": str(artifact_path),
            "repo_id": "user/repo",
            "repo_path": "runs/manual/artifact",
            "token": "token",
            "commit_message": "Manual artifact upload",
        }
    ]
