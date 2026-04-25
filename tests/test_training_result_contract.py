from optimizer_comparison.training.result_contract import build_training_result


# /**
#  * Проверяет верхнеуровневые поля общего training-result.
#  *
#  * @return None.
#  */
def test_training_result_has_required_top_level_keys() -> None:
    result = build_training_result(
        run_name="mock_adamw",
        status="completed",
        final_loss=0.0,
        training_time_seconds=0.0,
        max_memory_mb=0.0,
    )

    assert result["status"] == "completed"
    assert result["run_name"] == "mock_adamw"
    assert "metrics" in result
    assert "artifacts" in result


# /**
#  * Проверяет обязательные поля metrics в общем training-result.
#  *
#  * @return None.
#  */
def test_training_result_has_required_metrics() -> None:
    result = build_training_result(
        run_name="mock_adamw",
        status="completed",
        final_loss=0.0,
        training_time_seconds=0.0,
        max_memory_mb=0.0,
    )

    metrics = result["metrics"]

    assert isinstance(metrics, dict)
    assert metrics["final_loss"] == 0.0
    assert metrics["training_time_seconds"] == 0.0
    assert metrics["max_memory_mb"] == 0.0


# /**
#  * Проверяет обязательные поля artifacts для будущего сохранения модели и токенизатора.
#  *
#  * @return None.
#  */
def test_training_result_has_required_artifacts() -> None:
    result = build_training_result(
        run_name="mock_adamw",
        status="completed",
        final_loss=0.0,
        training_time_seconds=0.0,
        max_memory_mb=0.0,
    )

    artifacts = result["artifacts"]

    assert isinstance(artifacts, dict)
    assert "run_dir" in artifacts
    assert "config_path" in artifacts
    assert "result_path" in artifacts
    assert "model" in artifacts
    assert "tokenizer" in artifacts
    assert "checkpoints" in artifacts

    model_artifacts = artifacts["model"]

    assert isinstance(model_artifacts, dict)
    assert "local_path" in model_artifacts
    assert "hf_repo_id" in model_artifacts
    assert "hf_commit_url" in model_artifacts
    assert "upload_status" in model_artifacts
    assert "upload_error" in model_artifacts
