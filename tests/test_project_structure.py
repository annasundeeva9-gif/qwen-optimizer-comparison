from pathlib import Path


# /**
#  * Проверяет наличие основных директорий проектного каркаса.
#  *
#  * @return None.
#  */
def test_expected_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    expected_directories = [
        root / "configs",
        root / "scripts",
        root / "src" / "optimizer_comparison",
        root / "tests",
    ]

    for directory in expected_directories:
        assert directory.is_dir()


# /**
#  * Проверяет, что основной Python-пакет импортируется.
#  *
#  * @return None.
#  */
def test_package_can_be_imported() -> None:
    import optimizer_comparison

    assert optimizer_comparison.__version__
