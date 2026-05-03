import json
from pathlib import Path

from omegaconf import OmegaConf

from optimizer_comparison.evaluation.model_source import BASE_QWEN_MODEL_ID, BASE_QWEN_RUN_NAME
from optimizer_comparison.tracking.mlflow_logger import (
    collect_training_params,
    is_tracking_enabled,
    log_custom_artifacts,
    log_evaluation_run,
    log_hf_hub_tags,
    log_training_artifacts,
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
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "experiment": {"name": "mock_adamw"},
            "data": {"split": {"seed": 42}},
            "training": {
                "seed": 42,
                "data_seed": 43,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "max_grad_norm": 1.0,
                "max_steps": None,
            },
        }
    )

    params = collect_training_params(config)

    assert params["project.name"] == "optimizer-comparison"
    assert params["mode.name"] == "mock"
    assert params["model.name"] == "mock"
    assert params["optimizer.name"] == "adamw"
    assert params["optimizer.lr"] == "0.0001"
    assert params["optimizer.weight_decay"] == "0.01"
    assert params["optimizer.betas"] == "[0.9, 0.95]"
    assert params["optimizer.eps"] == "1e-08"
    assert params["experiment.name"] == "mock_adamw"
    assert params["data.split.seed"] == "42"
    assert params["training.seed"] == "42"
    assert params["training.data_seed"] == "43"
    assert params["training.lr_scheduler_type"] == "cosine"
    assert params["training.warmup_ratio"] == "0.03"
    assert params["training.max_grad_norm"] == "1.0"
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
                {"step": 1, "loss": 2.0, "learning_rate": 0.0001, "grad_norm": 0.3},
                {"step": 2, "eval_loss": 1.5},
                {"loss": 9.0},
            ]
        }
    )

    assert logged_metrics == [
        ("train/loss", 2.0, 1),
        ("train/learning_rate", 0.0001, 1),
        ("train/grad_norm", 0.3, 1),
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


# /**
#  * Проверяет, что custom artifacts логируются только явно переданными путями.
#  *
#  * @param monkeypatch Инструмент pytest для подмены mlflow.log_artifact.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_log_custom_artifacts_logs_explicit_files(monkeypatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "plot.png"
    artifact_path.write_text("plot", encoding="utf-8")
    logged_artifacts: list[str] = []

    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.log_artifact",
        lambda path: logged_artifacts.append(path),
    )

    log_custom_artifacts(
        config=OmegaConf.create({"tracking": {"log_artifacts": True}}),
        artifact_paths=[str(artifact_path), str(tmp_path / "missing.png")],
    )

    assert logged_artifacts == [str(artifact_path)]


# /**
#  * Проверяет логирование training plot artifacts в MLflow.
#  *
#  * @param monkeypatch Инструмент pytest для подмены mlflow.log_artifact.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_log_training_artifacts_logs_training_curves(monkeypatch, tmp_path: Path) -> None:
    plot_path = tmp_path / "training_curves.png"
    plot_path.write_bytes(b"png")
    logged_artifacts: list[str] = []

    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.log_artifact",
        lambda path: logged_artifacts.append(path),
    )

    log_training_artifacts(
        config=OmegaConf.create({"tracking": {"log_artifacts": True}}),
        training_result={
            "artifacts": {
                "plots": {
                    "training_curves_path": str(plot_path),
                }
            }
        },
    )

    assert logged_artifacts == [str(plot_path)]


# /**
#  * Проверяет логирование evaluation metrics в существующий MLflow run.
#  *
#  * @param monkeypatch Инструмент pytest для подмены MLflow API.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_log_evaluation_run_logs_metrics_to_existing_run(monkeypatch, tmp_path: Path) -> None:
    lm_eval_result_path = tmp_path / "lm_eval_results.json"
    summary_path = tmp_path / "evaluation_summary.csv"
    lm_eval_result_path.write_text(
        json.dumps(
            {
                "results": {
                    "piqa": {
                        "acc,none": 0.75,
                        "acc_stderr,none": 0.01,
                        "alias": "piqa",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text("task,metric,value\npiqa,acc,0.75\n", encoding="utf-8")
    started_runs: list[str] = []
    logged_metrics: list[tuple[str, float]] = []
    logged_tags: list[dict[str, str]] = []
    logged_artifacts: list[str] = []

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_start_run(run_id: str):
        started_runs.append(run_id)
        return FakeRun()

    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_tracking_uri",
        lambda uri: None,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_experiment",
        lambda name: None,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.start_run",
        fake_start_run,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.log_metric",
        lambda name, value: logged_metrics.append((name, value)),
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_tags",
        lambda tags: logged_tags.append(tags),
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.log_artifact",
        lambda path: logged_artifacts.append(path),
    )

    log_evaluation_run(
        config=OmegaConf.create(
            {
                "tracking": {
                    "enabled": True,
                    "tracking_uri": str(tmp_path / "mlruns"),
                    "experiment_name": "optimizer-comparison",
                }
            }
        ),
        evaluation_result={
            "status": "completed",
            "mlflow_run_id": "mlflow-run-id",
            "lm_eval_result_path": str(lm_eval_result_path),
            "raw_log_path": str(tmp_path / "stdout.txt"),
            "summary_path": str(summary_path),
        },
    )

    assert started_runs == ["mlflow-run-id"]
    assert logged_metrics == [
        ("eval_harness/piqa/acc/none", 0.75),
    ]
    assert logged_artifacts == [str(summary_path)]
    assert logged_tags == [
        {
            "evaluation.status": "completed",
            "evaluation.result_path": str(lm_eval_result_path),
            "evaluation.raw_log_path": str(tmp_path / "stdout.txt"),
        }
    ]


# /**
#  * Проверяет создание отдельного MLflow run для baseline evaluation без training run id.
#  *
#  * @param monkeypatch Инструмент pytest для подмены MLflow API.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_log_evaluation_run_creates_run_for_base_model(monkeypatch, tmp_path: Path) -> None:
    lm_eval_result_path = tmp_path / "lm_eval_results.json"
    lm_eval_result_path.write_text(
        json.dumps({"results": {"piqa": {"acc,none": 0.75}}}),
        encoding="utf-8",
    )
    started_run_names: list[str] = []
    logged_tags: list[dict[str, str]] = []

    class FakeRun:
        def __init__(self) -> None:
            self.info = type("Info", (), {"run_id": "new-base-run-id"})()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_start_run(run_name: str):
        started_run_names.append(run_name)
        return FakeRun()

    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_tracking_uri",
        lambda uri: None,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_experiment",
        lambda name: None,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.start_run",
        fake_start_run,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.log_metric",
        lambda name, value: None,
    )
    monkeypatch.setattr(
        "optimizer_comparison.tracking.mlflow_logger.mlflow.set_tags",
        lambda tags: logged_tags.append(tags),
    )

    mlflow_run_id = log_evaluation_run(
        config=OmegaConf.create(
            {
                "tracking": {
                    "enabled": True,
                    "tracking_uri": str(tmp_path / "mlruns"),
                    "experiment_name": "optimizer-comparison",
                }
            }
        ),
        evaluation_result={
            "status": "completed",
            "run_name": BASE_QWEN_RUN_NAME,
            "mlflow_run_id": None,
            "lm_eval_result_path": str(lm_eval_result_path),
            "raw_log_path": str(tmp_path / "stdout.txt"),
            "model_source": {
                "type": "base_model",
                "base_model_id": BASE_QWEN_MODEL_ID,
            },
        },
    )

    assert mlflow_run_id == "new-base-run-id"
    assert started_run_names == [BASE_QWEN_RUN_NAME]
    assert logged_tags == [
        {
            "evaluation.status": "completed",
            "evaluation.result_path": str(lm_eval_result_path),
            "evaluation.source_type": "base_model",
            "evaluation.base_model_id": BASE_QWEN_MODEL_ID,
            "evaluation.raw_log_path": str(tmp_path / "stdout.txt"),
        }
    ]
