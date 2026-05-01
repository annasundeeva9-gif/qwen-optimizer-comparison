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
    assert "history" in result
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
    assert "hf_hub" in artifacts

    model_artifacts = artifacts["model"]

    assert isinstance(model_artifacts, dict)
    assert "local_path" in model_artifacts
    assert "hf_repo_id" in model_artifacts
    assert "hf_commit_url" in model_artifacts
    assert "upload_status" in model_artifacts
    assert "upload_error" in model_artifacts

    checkpoint_artifacts = artifacts["checkpoints"]

    assert isinstance(checkpoint_artifacts, dict)
    assert "local_path" in checkpoint_artifacts
    assert "best_path" in checkpoint_artifacts
    assert "last_path" in checkpoint_artifacts
    assert "cleanup_status" in checkpoint_artifacts
    assert "cleanup_error" in checkpoint_artifacts
    assert "removed_paths" in checkpoint_artifacts

    hf_hub_artifacts = artifacts["hf_hub"]

    assert isinstance(hf_hub_artifacts, dict)
    assert "repo_id" in hf_hub_artifacts
    assert "artifact_path" in hf_hub_artifacts
    assert "revision" in hf_hub_artifacts
    assert "commit_url" in hf_hub_artifacts
    assert "result_commit_url" in hf_hub_artifacts
    assert "upload_status" in hf_hub_artifacts
    assert "upload_error" in hf_hub_artifacts
