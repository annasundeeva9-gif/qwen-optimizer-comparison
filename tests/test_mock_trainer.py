from omegaconf import OmegaConf

from optimizer_comparison.training.mock_trainer import run_mock_training


# /**
#  * Проверяет, что mock trainer возвращает общий training-result контракт.
#  *
#  * @return None.
#  */
def test_mock_trainer_returns_training_result_contract() -> None:
    config = OmegaConf.create({"experiment": {"name": "mock_adamw"}})

    result = run_mock_training(config)

    assert result["status"] == "completed"
    assert result["run_name"] == "mock_adamw"
    assert "metrics" in result
    assert "artifacts" in result

    metrics = result["metrics"]
    artifacts = result["artifacts"]

    assert isinstance(metrics, dict)
    assert metrics["final_loss"] == 0.0
    assert metrics["training_time_seconds"] == 0.0
    assert metrics["max_memory_mb"] == 0.0

    assert isinstance(artifacts, dict)
    assert "model" in artifacts
    assert "tokenizer" in artifacts
    assert "checkpoints" in artifacts
