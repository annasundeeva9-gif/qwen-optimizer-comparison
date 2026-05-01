from omegaconf import OmegaConf

from optimizer_comparison.tracking.mlflow_logger import (
    collect_training_params,
    is_tracking_enabled,
    log_hf_hub_tags,
    log_training_history,
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
                "name": "mock",
                "pretrained_name_or_path": None,
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
    assert params["model.name"] == "mock"
    assert params["optimizer.name"] == "adamw"
    assert params["experiment.name"] == "mock_adamw"
    assert params["data.split.seed"] == "42"
    assert "training.max_steps" not in params


# /**
#  * Проверяет логирование step history в MLflow.
#  *
#  * @param monkeypatch Инструмент pytest для подмены mlflow.log_metric.
#  * @return None.
#  */
def test_log_training_history_logs_step_metrics(monkeypatch) -> None:
    logged_metrics: list[tuple[str, float, int | None]] = []

    def fake_log_metric(name: str, value: float, step: int | None = None) -> None:
        logged_metrics.append((name, value, step))

    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.log_metric",
        fake_log_metric,
    )
    log_training_history(
        {
            "history": [
                {"step": 1, "loss": 2.0, "learning_rate": 0.0001},
                {"step": 2, "eval_loss": 1.5},
                {"loss": 9.0},
            ]
        }
    )

    assert logged_metrics == [
        ("train/loss", 2.0, 1),
        ("train/learning_rate", 0.0001, 1),
        ("eval/loss", 1.5, 2),
    ]


# /**
#  * Проверяет логирование HF Hub metadata в MLflow tags.
#  *
#  * @param monkeypatch Инструмент pytest для подмены mlflow.set_tags.
#  * @return None.
#  */
def test_log_hf_hub_tags_logs_artifact_metadata(monkeypatch) -> None:
    logged_tags: list[dict[str, str]] = []

    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_tags",
        lambda tags: logged_tags.append(tags),
    )

    log_hf_hub_tags(
        {
            "artifacts": {
                "hf_hub": {
                    "repo_id": "user/repo",
                    "commit_url": "https://huggingface.co/user/repo/commit/abc",
                    "revision": "abc",
                    "artifact_path": "runs/adamw/run",
                    "upload_error": None,
                }
            }
        }
    )

    assert logged_tags == [
        {
            "hf_hub.repo_id": "user/repo",
            "hf_hub.commit_url": "https://huggingface.co/user/repo/commit/abc",
            "hf_hub.revision": "abc",
            "hf_hub.artifact_path": "runs/adamw/run",
        }
    ]
