from omegaconf import OmegaConf

from optimizer_comparison.tracking.mlflow_logger import (
    collect_training_params,
    is_tracking_enabled,
)


# /**
#  * Проверяет чтение флага включения MLflow tracking.
#  *
#  * @return None.
#  */
def test_is_tracking_enabled_reads_tracking_section() -> None:
    config = OmegaConf.create({"tracking": {"enabled": True}})

    assert is_tracking_enabled(config)


# /**
#  * Проверяет выбор компактных training-параметров для MLflow.
#  *
#  * @return None.
#  */
def test_collect_training_params_returns_expected_keys() -> None:
    config = OmegaConf.create(
        {
            "project": {"name": "optimizer-comparison"},
            "mode": {"name": "mock"},
            "model": {
                "name": "tiny",
                "pretrained_name_or_path": "sshleifer/tiny-gpt2",
            },
            "optimizer": {"name": "adamw"},
            "experiment": {"name": "mock_adamw"},
            "data": {"split": {"seed": 42}},
            "training": {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": None,
            },
        }
    )

    params = collect_training_params(config)

    assert params["project.name"] == "optimizer-comparison"
    assert params["mode.name"] == "mock"
    assert params["model.name"] == "tiny"
    assert params["optimizer.name"] == "adamw"
    assert params["experiment.name"] == "mock_adamw"
    assert params["data.split.seed"] == "42"
    assert "training.max_steps" not in params
