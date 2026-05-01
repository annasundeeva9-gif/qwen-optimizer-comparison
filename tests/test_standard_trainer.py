from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from optimizer_comparison.training import trainer as trainer_module
from optimizer_comparison.training.trainer import build_training_arguments, run_training


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
        self.state = SimpleNamespace(
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
    def train(self) -> FakeTrainOutput:
        self.train_called = True
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
            "optimizer": {"name": "adamw"},
            "data": {"final": {"dir": "unused"}},
            "model": {"name": "tiny_qwen_2_5"},
            "training": {
                "num_train_epochs": 1,
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

    monkeypatch.setattr(
        trainer_module,
        "load_final_training_dataset",
        lambda config: make_dataset(),
    )
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
#  * Проверяет, что standard trainer path явно отклоняет не-AdamW optimizer.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_training_rejects_non_adamw_optimizer(tmp_path: Path) -> None:
    config = make_config()
    config.optimizer.name = "muon"

    with pytest.raises(NotImplementedError, match="MuonTrainer"):
        run_training(config=config, run_dir=tmp_path)


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
