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

