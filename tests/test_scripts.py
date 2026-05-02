from pathlib import Path


# /**
#  * Проверяет, что mock shell-скрипт запускает правильный Hydra override набор.
#  *
#  * @return None.
#  */
def test_run_mock_script_contains_expected_command() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_mock.sh").read_text(encoding="utf-8")

    assert "python -m optimizer_comparison.train" in script
    assert "mode=mock" in script
    assert "model=mock" in script
    assert "optimizer=adamw" in script
    assert "experiment=mock_adamw" in script


# /**
#  * Проверяет, что mock shell-скрипт выставляет PYTHONPATH для запуска без editable install.
#  *
#  * @return None.
#  */
def test_run_mock_script_sets_project_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_mock.sh").read_text(encoding="utf-8")

    assert 'PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in script
    assert 'export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"' in script


# /**
#  * Проверяет, что smoke shell-скрипт запускает tiny Qwen2.5 model config.
#  *
#  * @return None.
#  */
def test_run_smoke_script_uses_tiny_qwen_model() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_smoke.sh").read_text(encoding="utf-8")

    assert "python -m optimizer_comparison.train" in script
    assert "mode=smoke" in script
    assert "model=tiny_qwen_2_5" in script
    assert "optimizer=adamw" in script
    assert "experiment=smoke_adamw_tiny" in script


# /**
#  * Проверяет, что smoke shell-скрипт выставляет PYTHONPATH для запуска без editable install.
#  *
#  * @return None.
#  */
def test_run_smoke_script_sets_project_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_smoke.sh").read_text(encoding="utf-8")

    assert 'PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in script
    assert 'export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"' in script


# /**
#  * Проверяет, что eval shell-скрипт требует run_dir и запускает evaluate.
#  *
#  * @return None.
#  */
def test_run_eval_script_uses_run_dir_source() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_eval.sh").read_text(encoding="utf-8")

    assert "Usage: $0 <run_dir>" in script
    assert "python -m optimizer_comparison.evaluate" in script
    assert '"evaluation.source.run_dir=${RUN_DIR}"' in script


# /**
#  * Проверяет, что train+eval shell-скрипт запускает обучение и затем evaluation.
#  *
#  * @return None.
#  */
def test_run_train_eval_script_runs_train_then_eval() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_train_eval.sh").read_text(encoding="utf-8")

    assert "python -m optimizer_comparison.train" in script
    assert "RUN_DIR=\"$(ls -td " in script
    assert "python -m optimizer_comparison.evaluate" in script
    assert '"evaluation.source.run_dir=${RUN_DIR}"' in script


# /**
#  * Проверяет, что train grid содержит ручной список запусков и upload MLflow snapshot-а.
#  *
#  * @return None.
#  */
def test_run_train_grid_script_uploads_mlflow_snapshot() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_train_grid.sh").read_text(encoding="utf-8")

    assert "RUNS=(" in script
    assert "optimizer=adamw experiment=adamw_baseline" in script
    assert "optimizer=muon experiment=muon_baseline" in script
    assert "python scripts/upload_mlflow_snapshot.py" in script
    assert "HF_REPO_ID" in script


# /**
#  * Проверяет, что eval grid содержит ручной список run_dir-ов.
#  *
#  * @return None.
#  */
def test_run_eval_grid_script_uses_manual_run_dirs() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_eval_grid.sh").read_text(encoding="utf-8")

    assert "RUN_DIRS=(" in script
    assert "REPLACE_ME" in script
    assert "python -m optimizer_comparison.evaluate" in script
    assert '"evaluation.source.run_dir=${run_dir}"' in script
