from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from optimizer_comparison.train import resolve_training_run_dir


# /**
#  * Создает минимальный config для проверки выбора run directory.
#  *
#  * @param resume_from_run_dir Путь к run directory для resume или None.
#  * @return Hydra-like config.
#  */
def make_config(resume_from_run_dir: str | None) -> DictConfig:
    return OmegaConf.create({"training": {"resume_from_run_dir": resume_from_run_dir}})


# /**
#  * Проверяет, что без resume создается новый run directory.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_training_run_dir_creates_new_run(tmp_path: Path) -> None:
    run_dir = resolve_training_run_dir(
        config=make_config(resume_from_run_dir=None),
        artifacts_root_dir=str(tmp_path),
        experiment_name="adamw_baseline",
    )

    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("adamw_baseline__")
    assert run_dir.is_dir()


# /**
#  * Проверяет, что resume использует существующую директорию run-а.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_training_run_dir_uses_existing_resume_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "adamw_baseline__2026-05-02_10-00-00"
    run_dir.mkdir()

    resolved_run_dir = resolve_training_run_dir(
        config=make_config(resume_from_run_dir=str(run_dir)),
        artifacts_root_dir=str(tmp_path / "unused"),
        experiment_name="adamw_baseline",
    )

    assert resolved_run_dir == run_dir


# /**
#  * Проверяет понятную ошибку, если resume directory отсутствует.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_resolve_training_run_dir_rejects_missing_resume_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Resume run directory"):
        resolve_training_run_dir(
            config=make_config(resume_from_run_dir=str(tmp_path / "missing")),
            artifacts_root_dir=str(tmp_path),
            experiment_name="adamw_baseline",
        )
