from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from optimizer_comparison.training import training_loop as trainer_module
from optimizer_comparison.training.training_loop import (
    build_trainer,
    build_training_arguments,
    run_training,
)


# /**
#  * Fake train output с training_loss, совместимый с run_training.
#  */
class FakeTrainOutput:
    training_loss = 1.25


# /**
#  * Fake trainer для проверки orchestration без реального обучения.
#  */
class FakeTrainer:
    # /**
    #  * Создает fake trainer с минимальным train() методом.
    #  */
    def __init__(self) -> None:
        self.train_called = False
        self.resume_from_checkpoint = None
        self.state = SimpleNamespace(
            global_step=2,
            log_history=[
                {"step": 1, "loss": 2.0, "learning_rate": 1e-5},
                {"step": 2, "eval_loss": 1.5},
            ]
        )

    # /**
    #  * Имитирует Trainer.train().
    #  *
    #  * @return Fake train output.
    #  */
    def train(self, resume_from_checkpoint: str | None = None) -> FakeTrainOutput:
        self.train_called = True
        self.resume_from_checkpoint = resume_from_checkpoint
        return FakeTrainOutput()


# /**
#  * Fake tokenizer с save_pretrained для fallback-сценариев.
#  */
class FakeTokenizer:
    # /**
    #  * Имитирует сохранение tokenizer.
    #  *
    #  * @param output_dir Целевая директория.
    #  * @return None.
    #  */
    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


# /**
#  * Создает минимальный полный конфиг для standard trainer path.
#  *
#  * @return Hydra-like конфиг.
#  */
def make_config() -> Any:
    return OmegaConf.create(
        {
            "experiment": {"name": "smoke_adamw_tiny"},
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "data": {"final": {"dir": "unused"}},
            "model": {"name": "tiny_qwen_2_5"},
            "training": {
                "num_train_epochs": 1,
                "seed": 42,
                "data_seed": 43,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": None,
                "logging_strategy": "steps",
                "logging_steps": 1,
                "eval_strategy": "steps",
                "eval_steps": 1,
                "save_strategy": "steps",
                "save_steps": 1,
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "use_gradient_checkpointing": False,
                "bf16": False,
                "fp16": False,
                "dataloader_num_workers": 0,
                "remove_unused_columns": False,
                "report_to": [],
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "max_grad_norm": 1.0,
                "checkpoints": {
                    "best_dir_name": "best",
                    "last_dir_name": "last",
                },
            },
        }
    )


# /**
#  * Создает минимальный final DatasetDict для fake standard trainer.
#  *
#  * @return DatasetDict с train и validation split-ами.
#  */
def make_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1]], "attention_mask": [[1]]}),
            "validation": Dataset.from_dict({"input_ids": [[1]], "attention_mask": [[1]]}),
        }
    )


# /**
#  * Проверяет, что TrainingArguments строятся из Hydra-конфига и run directory.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_training_arguments_uses_run_directory(tmp_path: Path) -> None:
    args = build_training_arguments(config=make_config(), run_dir=tmp_path)

    assert args.output_dir == str(tmp_path / "trainer_output")
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 0.01
    assert args.adam_beta1 == 0.9
    assert args.adam_beta2 == 0.95
    assert args.adam_epsilon == 1e-8
    assert args.lr_scheduler_type.value == "cosine"
    assert args.warmup_ratio == 0.03
    assert args.max_grad_norm == 1.0
    assert args.seed == 42
    assert args.data_seed == 43
    assert args.per_device_train_batch_size == 1
    assert args.eval_strategy.value == "steps"
    assert args.save_strategy.value == "steps"
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"


# /**
#  * Проверяет, что run_training собирает результат и пути артефактов без реального Trainer.
#  *
#  * @param monkeypatch Инструмент pytest для подмены зависимостей.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_training_returns_training_result_with_artifacts(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_trainer = FakeTrainer()

    monkeypatch.setattr(trainer_module, "get_final_training_dataset", lambda config: make_dataset())
    monkeypatch.setattr(trainer_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(trainer_module, "build_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(trainer_module, "build_model", lambda config: object())
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        trainer_module.torch.cuda,
        "max_memory_allocated",
        lambda: int(12.5 * 1024 * 1024),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_standard_trainer",
        lambda **kwargs: fake_trainer,
    )
    monkeypatch.setattr(
        trainer_module,
        "save_best_and_last_checkpoints",
        lambda **kwargs: {
            "local_path": str(tmp_path / "checkpoints"),
            "best_path": str(tmp_path / "checkpoints" / "best"),
            "last_path": str(tmp_path / "checkpoints" / "last"),
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "save_final_model_artifacts",
        lambda **kwargs: {
            "model_path": str(tmp_path / "model"),
            "tokenizer_path": str(tmp_path / "tokenizer"),
        },
    )

    result = run_training(config=make_config(), run_dir=tmp_path)

    assert fake_trainer.train_called
    assert result["status"] == "completed"
    assert result["run_name"] == "smoke_adamw_tiny"
    assert result["metrics"]["final_loss"] == 1.25
    assert result["metrics"]["max_memory_mb"] == 12.5
    assert isinstance(result["metrics"]["time_per_step_seconds"], float)
    assert result["metrics"]["time_per_step_seconds"] > 0
    assert result["history"] == fake_trainer.state.log_history
    assert result["artifacts"]["model"]["local_path"] == str(tmp_path / "model")
    assert result["artifacts"]["tokenizer"]["local_path"] == str(tmp_path / "tokenizer")
    assert result["artifacts"]["checkpoints"]["best_path"] == str(
        tmp_path / "checkpoints" / "best"
    )
    assert result["artifacts"]["checkpoints"]["last_path"] == str(
        tmp_path / "checkpoints" / "last"
    )


# /**
#  * Проверяет, что Muon trainer path явно ожидает ручную интеграцию.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_trainer_rejects_pending_muon_optimizer(tmp_path: Path) -> None:
    config = make_config()
    config.optimizer.name = "muon"

    with pytest.raises(NotImplementedError, match="Muon trainer"):
        build_trainer(
            config=config,
            model=object(),
            tokenizer=FakeTokenizer(),
            dataset=make_dataset(),
            run_dir=tmp_path,
        )


# /**
#  * Проверяет, что настоящий training не запускается без CUDA.
#  *
#  * @param monkeypatch Инструмент pytest для подмены torch.cuda.is_available.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_training_requires_cuda(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_training(config=make_config(), run_dir=tmp_path)


# /**
#  * Проверяет, что run_training продолжает обучение из checkpoint-а существующего run-а.
#  *
#  * @param monkeypatch Инструмент pytest для подмены зависимостей.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_training_resumes_from_existing_checkpoint(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_trainer = FakeTrainer()
    checkpoint_dir = tmp_path / "trainer_output" / "checkpoint-7"
    checkpoint_dir.mkdir(parents=True)
    config = make_config()
    config.training.resume_from_run_dir = str(tmp_path)

    monkeypatch.setattr(trainer_module, "get_final_training_dataset", lambda config: make_dataset())
    monkeypatch.setattr(trainer_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(trainer_module, "build_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(trainer_module, "build_model", lambda config: object())
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        trainer_module.torch.cuda,
        "max_memory_allocated",
        lambda: int(12.5 * 1024 * 1024),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_standard_trainer",
        lambda **kwargs: fake_trainer,
    )
    monkeypatch.setattr(
        trainer_module,
        "save_best_and_last_checkpoints",
        lambda **kwargs: {
            "local_path": str(tmp_path / "checkpoints"),
            "best_path": str(tmp_path / "checkpoints" / "best"),
            "last_path": str(tmp_path / "checkpoints" / "last"),
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "save_final_model_artifacts",
        lambda **kwargs: {
            "model_path": str(tmp_path / "model"),
            "tokenizer_path": str(tmp_path / "tokenizer"),
        },
    )

    run_training(config=config, run_dir=tmp_path)

    assert fake_trainer.resume_from_checkpoint == str(checkpoint_dir)
