from pathlib import Path

from omegaconf import OmegaConf

from optimizer_comparison.artifacts.local_store import (
    build_run_id,
    create_run_dir,
    save_json,
    save_resolved_config,
)


# /**
#  * Проверяет создание run id по имени эксперимента и timestamp.
#  *
#  * @return None.
#  */
def test_build_run_id_uses_experiment_name_and_timestamp() -> None:
    assert (
        build_run_id(
            experiment_name="mock_adamw",
            timestamp="2026-04-25_20-30-00",
        )
        == "mock_adamw__2026-04-25_20-30-00"
    )


# /**
#  * Проверяет создание директории запуска по run id.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_create_run_dir_uses_run_id(tmp_path: Path) -> None:
    run_dir = create_run_dir(
        artifacts_root_dir=tmp_path,
        experiment_name="mock_adamw",
        timestamp="2026-04-25_20-30-00",
    )

    assert run_dir == tmp_path / "mock_adamw__2026-04-25_20-30-00"
    assert run_dir.is_dir()


# /**
#  * Проверяет сохранение resolved Hydra-конфига в директорию запуска.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_save_resolved_config_writes_yaml(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "project": {"name": "optimizer-comparison"},
            "experiment": {"name": "mock_adamw"},
            "value": "${experiment.name}",
        }
    )
    run_dir = create_run_dir(
        artifacts_root_dir=tmp_path,
        experiment_name="mock_adamw",
        timestamp="2026-04-25_20-30-00",
    )

    config_path = save_resolved_config(config=config, run_dir=run_dir)

    assert config_path == run_dir / "config.yaml"
    assert config_path.is_file()
    assert "value: mock_adamw" in config_path.read_text(encoding="utf-8")


# /**
#  * Проверяет сохранение словаря в JSON-файл.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_save_json_writes_file(tmp_path: Path) -> None:
    output_path = save_json(
        data={"status": "completed", "metrics": {"final_loss": 0.0}},
        output_path=tmp_path / "result.json",
    )

    assert output_path.is_file()
    assert '"status": "completed"' in output_path.read_text(encoding="utf-8")
