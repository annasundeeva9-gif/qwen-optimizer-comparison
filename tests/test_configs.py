from pathlib import Path

from omegaconf import OmegaConf


# /**
#  * Проверяет, что mock model config не указывает на реальный HuggingFace артефакт.
#  *
#  * @return None.
#  */
def test_mock_model_config_is_explicit_placeholder() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "model" / "mock.yaml")

    assert config.name == "mock"
    assert config.pretrained_name_or_path is None
    assert config.tokenizer_name_or_path is None
    assert config.torch_dtype is None


# /**
#  * Проверяет, что smoke model config указывает на tiny Qwen2.5 из HuggingFace.
#  *
#  * @return None.
#  */
def test_tiny_qwen_model_config_uses_expected_hf_repo() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "model" / "tiny_qwen_2_5.yaml")

    expected_repo = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    assert config.name == "tiny_qwen_2_5"
    assert config.pretrained_name_or_path == expected_repo
    assert config.tokenizer_name_or_path == expected_repo
    assert config.trust_remote_code is False


# /**
#  * Проверяет, что training config содержит базовые настройки HF Trainer без отдельной группы.
#  *
#  * @return None.
#  */
def test_training_config_contains_trainer_and_checkpoint_blocks() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "training" / "default.yaml")

    assert config.trainer.name == "hf_standard"
    assert config.trainer.use_custom_optimizer is False
    assert config.checkpoints.save_best is True
    assert config.checkpoints.save_last is True
    assert config.checkpoints.best_dir_name == "best"
    assert config.checkpoints.last_dir_name == "last"
    assert config.load_best_model_at_end is True
    assert config.metric_for_best_model == "eval_loss"
    assert config.eval_strategy == "steps"
    assert config.save_strategy == "steps"
    assert config.logging_strategy == "steps"
