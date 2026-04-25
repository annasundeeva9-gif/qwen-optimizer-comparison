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
    assert "model=tiny" in script
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
