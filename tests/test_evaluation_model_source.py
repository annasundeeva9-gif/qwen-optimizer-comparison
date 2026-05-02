from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from optimizer_comparison.evaluation import model_source as model_source_module
from optimizer_comparison.evaluation.model_source import resolve_evaluation_model_source


# /**
#  * Создает evaluation config с local source.
#  *
#  * @param model_path Путь к директории модели.
#  * @param tokenizer_path Путь к директории токенизатора.
#  * @return Hydra-like config.
#  */
def make_local_config(
    model_path: Path | None,
    tokenizer_path: Path | None,
    run_dir: Path | None = None,
) -> Any:
    return OmegaConf.create(
        {
            "evaluation": {
                "source": {
                    "use_hf_hub": False,
                    "run_dir": None if run_dir is None else str(run_dir),
                    "model_path": None if model_path is None else str(model_path),
                    "tokenizer_path": None if tokenizer_path is None else str(tokenizer_path),
                }
            }
        }
    )


# /**
#  * Создает evaluation config с HF Hub source.
#  *
#  * @param download_dir Локальная директория для скачивания artifacts.
#  * @return Hydra-like config.
#  */
def make_hf_hub_config(download_dir: Path) -> Any:
    return OmegaConf.create(
        {
            "evaluation": {
                "source": {
                    "use_hf_hub": True,
                    "repo_id": "user/repo",
                    "repo_path": "runs/adamw_baseline/2026-05-01_10-00-00",
                    "revision": "abc123",
                    "token_env_var": "HF_TOKEN",
                    "download_dir": str(download_dir),
                }
            }
        }
    )


# /**
#  * Проверяет local source для существующих model/tokenizer директорий.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_evaluation_model_source_uses_local_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    tokenizer_dir = tmp_path / "tokenizer"
    model_dir.mkdir()
    tokenizer_dir.mkdir()

    source = resolve_evaluation_model_source(
        make_local_config(model_path=model_dir, tokenizer_path=tokenizer_dir)
    )

    assert source.source_type == "local"
    assert source.run_dir == tmp_path
    assert source.model_path == model_dir
    assert source.tokenizer_path == tokenizer_dir
    assert source.hf_repo_id is None


# /**
#  * Проверяет понятную ошибку для отсутствующего local model path.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_evaluation_model_source_requires_existing_local_model(
    tmp_path: Path,
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="model_path"):
        resolve_evaluation_model_source(
            make_local_config(model_path=tmp_path / "missing_model", tokenizer_path=tokenizer_dir)
        )


# /**
#  * Проверяет local source через training run directory.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_evaluation_model_source_uses_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "adamw_baseline__2026-05-01_10-00-00"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "tokenizer").mkdir()

    source = resolve_evaluation_model_source(
        make_local_config(model_path=None, tokenizer_path=None, run_dir=run_dir)
    )

    assert source.source_type == "local"
    assert source.run_dir == run_dir
    assert source.model_path == run_dir / "model"
    assert source.tokenizer_path == run_dir / "tokenizer"


# /**
#  * Проверяет восстановление model/tokenizer paths из HF Hub artifacts.
#  *
#  * @param monkeypatch Инструмент pytest для подмены download helper.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_evaluation_model_source_downloads_hf_hub_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    restored_dir = tmp_path / "downloaded" / "runs" / "adamw_baseline" / "2026-05-01_10-00-00"
    (restored_dir / "model").mkdir(parents=True)
    (restored_dir / "tokenizer").mkdir()

    def fake_download_artifacts_from_hf(**kwargs: object) -> Path:
        calls.append(kwargs)
        return restored_dir

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(
        model_source_module,
        "download_artifacts_from_hf",
        fake_download_artifacts_from_hf,
    )

    source = resolve_evaluation_model_source(make_hf_hub_config(tmp_path / "downloaded"))

    assert source.source_type == "hf_hub"
    assert source.run_dir == restored_dir
    assert source.model_path == restored_dir / "model"
    assert source.tokenizer_path == restored_dir / "tokenizer"
    assert source.hf_repo_id == "user/repo"
    assert source.hf_artifact_path == "runs/adamw_baseline/2026-05-01_10-00-00"
    assert source.hf_revision == "abc123"
    assert calls == [
        {
            "repo_id": "user/repo",
            "target_dir": tmp_path / "downloaded",
            "repo_path": "runs/adamw_baseline/2026-05-01_10-00-00",
            "token": "token",
            "revision": "abc123",
        }
    ]


# /**
#  * Проверяет понятную ошибку, если в скачанных artifacts нет tokenizer директории.
#  *
#  * @param monkeypatch Инструмент pytest для подмены download helper.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_evaluation_model_source_requires_downloaded_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restored_dir = tmp_path / "downloaded" / "run"
    (restored_dir / "model").mkdir(parents=True)

    def fake_download_artifacts_from_hf(**kwargs: object) -> Path:
        return restored_dir

    monkeypatch.setattr(
        model_source_module,
        "download_artifacts_from_hf",
        fake_download_artifacts_from_hf,
    )

    with pytest.raises(FileNotFoundError, match="tokenizer"):
        resolve_evaluation_model_source(make_hf_hub_config(tmp_path / "downloaded"))
