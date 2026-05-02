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


# /**
#  * Проверяет базовый контракт lm-evaluation-harness конфига.
#  *
#  * @return None.
#  */
def test_evaluation_config_contains_lm_eval_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "evaluation" / "lm_eval.yaml")

    assert list(config.harness.tasks) == [
        "piqa",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "hellaswag",
    ]
    assert config.harness.batch_size == "auto"
    assert "device" not in config.harness
    assert config.harness.output_path == "outputs/eval/lm_eval_results.json"
    assert config.harness.raw_log_path == "outputs/eval/lm_eval_stdout.txt"
    assert config.harness.result_filename == "evaluation_result.json"

    assert config.source.use_hf_hub is False
    assert config.source.run_dir is None
    assert config.source.model_path is None
    assert config.source.tokenizer_path is None
    assert config.source.repo_id is None
    assert config.source.repo_path is None
    assert config.source.revision is None
    assert config.source.token_env_var == "HF_TOKEN"
    assert config.source.download_dir == "outputs/eval/hf_artifacts"


# /**
#  * Проверяет лимиты evaluation samples для smoke/full режимов.
#  *
#  * @return None.
#  */
def test_mode_configs_define_expected_evaluation_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    smoke_config = OmegaConf.load(root / "configs" / "mode" / "smoke.yaml")
    full_config = OmegaConf.load(root / "configs" / "mode" / "full.yaml")

    assert smoke_config.limit_eval_samples == 32
    assert smoke_config.data.split.dir == "outputs/datasets/split_raw/openwebtext_100k_smoke"
    assert smoke_config.data.tokenization.dir == (
        "outputs/datasets/tokenized/openwebtext_100k_smoke"
    )
    assert smoke_config.data.final.dir == "outputs/datasets/final/openwebtext_100k_smoke"
    assert full_config.limit_eval_samples is None
