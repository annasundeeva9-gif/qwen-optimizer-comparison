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
#  * Проверяет, что training configs содержат общий protocol без выбора trainer path.
#  *
#  * @return None.
#  */
def test_training_configs_contain_shared_protocol_without_trainer_switch() -> None:
    root = Path(__file__).resolve().parents[1]
    smoke_config = OmegaConf.load(root / "configs" / "training" / "smoke.yaml")
    full_config = OmegaConf.load(root / "configs" / "training" / "full.yaml")

    for config in (smoke_config, full_config):
        assert "trainer" not in config
        assert config.checkpoints.best_dir_name == "best"
        assert config.checkpoints.last_dir_name == "last"
        assert config.seed == 42
        assert config.data_seed == 42
        assert config.load_best_model_at_end is True
        assert config.metric_for_best_model == "eval_loss"
        assert config.eval_strategy == "steps"
        assert config.save_strategy == "steps"
        assert config.logging_strategy == "steps"
        assert "lr_scheduler_type" in config
        assert "warmup_ratio" in config
        assert "max_grad_norm" in config


# /**
#  * Проверяет placeholder config для будущего combined optimizer path.
#  *
#  * @return None.
#  */
def test_combined_optimizer_config_is_pending_manual_integration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "optimizer" / "combined.yaml")

    assert config.name == "combined"
    assert config.implementation == "pending_manual_integration"


# /**
#  * Проверяет smoke experiment config для Muon.
#  *
#  * @return None.
#  */
def test_smoke_muon_experiment_config_uses_muon_tag() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "experiment" / "smoke_muon_tiny.yaml")

    assert config.name == "smoke_muon_tiny"
    assert config.tags.stage == "smoke"
    assert config.tags.optimizer == "muon"
    assert config.tags.model == "tiny_qwen_2_5"


# /**
#  * Проверяет текущий контракт конфигурации Muon optimizer.
#  *
#  * @return None.
#  */
def test_muon_optimizer_config_contains_parameters() -> None:
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "optimizer" / "muon.yaml")

    assert config.name == "muon"
    assert config.implementation == "custom"
    assert config.lr == 1e-4
    assert config.weight_decay == 0.01
    assert config.muon_layer_count == 24
    assert list(config.muon_param_patterns) == [
        "model.layers.{layer}.self_attn.q_proj.weight",
        "model.layers.{layer}.self_attn.k_proj.weight",
        "model.layers.{layer}.self_attn.v_proj.weight",
        "model.layers.{layer}.self_attn.o_proj.weight",
        "model.layers.{layer}.mlp.gate_proj.weight",
        "model.layers.{layer}.mlp.up_proj.weight",
        "model.layers.{layer}.mlp.down_proj.weight",
    ]
    assert config.momentum == 0.95
    assert config.nesterov is True
    assert config.ns_steps == 5
    assert list(config.adamw_betas) == [0.9, 0.95]
    assert config.adamw_eps == 1e-8


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
    assert isinstance(config.harness.batch_size, int)
    assert config.harness.batch_size > 0
    assert "device" not in config.harness
    assert config.harness.output_path == "outputs/eval/lm_eval_results.json"
    assert config.harness.raw_log_path == "outputs/eval/lm_eval_stdout.txt"
    assert config.harness.result_filename == "evaluation_result.json"

    assert config.source.use_base_model is False
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
