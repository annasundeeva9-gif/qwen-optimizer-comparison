"""Training loop implementation."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from optimizer_comparison.data.dataset_loader import (
    dataset_local_path_exists,
    load_dataset_local,
)
from optimizer_comparison.data.prepare_data import run_data_pipeline
from optimizer_comparison.models.build_model import build_model
from optimizer_comparison.models.tokenization import build_tokenizer
from optimizer_comparison.training.result_contract import TrainingResult, build_training_result
from optimizer_comparison.training.seed import set_seed

REQUIRED_FINAL_SPLITS: tuple[str, ...] = ("train", "validation")
REQUIRED_FINAL_COLUMNS: tuple[str, ...] = ("input_ids", "attention_mask")


# /**
#  * Возвращает последний checkpoint Trainer-а с явным типом для strict mypy.
#  *
#  * @param output_dir Директория output_dir из TrainingArguments.
#  * @return Путь к последнему checkpoint-у или None, если checkpoint не найден.
#  */
def find_last_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoint = get_last_checkpoint(str(output_dir))  # type: ignore[no-untyped-call]
    if checkpoint is None:
        return None
    return str(checkpoint)


# /**
#  * Возвращает checkpoint для resume существующего training run-а.
#  *
#  * @param run_dir Директория существующего run-а.
#  * @return Путь к последнему checkpoint-у в trainer_output.
#  */
def find_resume_checkpoint(run_dir: str | Path) -> str:
    output_dir = Path(run_dir) / "trainer_output"
    checkpoint = find_last_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No trainer checkpoint found for resume: {output_dir}")
    return checkpoint


# /**
#  * Возвращает best checkpoint Trainer-а с явным типом для strict mypy.
#  *
#  * @param trainer HuggingFace Trainer после завершения train().
#  * @return Путь к best checkpoint-у или None, если Trainer его не выбрал.
#  */
def get_best_checkpoint(trainer: Trainer) -> str | None:
    checkpoint = trainer.state.best_model_checkpoint
    if checkpoint is None:
        return None
    return checkpoint


# /**
#  * Проверяет один final split перед передачей в Trainer.
#  *
#  * @param dataset Split HuggingFace Dataset.
#  * @param split_name Имя split-а для понятного текста ошибки.
#  * @return None.
#  */
def validate_final_training_split(dataset: Dataset, split_name: str) -> None:
    missing_columns = [
        column_name
        for column_name in REQUIRED_FINAL_COLUMNS
        if column_name not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"Final training dataset split '{split_name}' is missing columns: {missing_columns}"
        )


# /**
#  * Проверяет, что final dataset имеет split-ы и колонки, нужные training loop.
#  *
#  * @param dataset Загруженный HuggingFace DatasetDict.
#  * @return None.
#  */
def validate_final_training_dataset(dataset: DatasetDict) -> None:
    missing_splits = [
        split_name for split_name in REQUIRED_FINAL_SPLITS if split_name not in dataset
    ]
    if missing_splits:
        raise ValueError(f"Final training dataset is missing splits: {missing_splits}")

    for split_name in REQUIRED_FINAL_SPLITS:
        validate_final_training_split(dataset=dataset[split_name], split_name=split_name)


# /**
#  * Возвращает final DatasetDict, при необходимости запуская data pipeline.
#  *
#  * @param config Полный Hydra-конфиг training run-а.
#  * @return DatasetDict с train и validation split-ами.
#  */
def get_final_training_dataset(config: DictConfig) -> DatasetDict:
    run_data_pipeline(config)
    final_dir = str(config.data.final.dir)
    if not dataset_local_path_exists(final_dir):
        raise FileNotFoundError(f"Final training dataset directory does not exist: {final_dir}")

    dataset = load_dataset_local(final_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Final training dataset must be a HuggingFace DatasetDict.")

    validate_final_training_dataset(dataset)
    return dataset

# /**
#  * Собирает TrainingArguments для стандартного HuggingFace Trainer.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @param run_dir Директория конкретного запуска.
#  * @return TrainingArguments для стандартного AdamW training path.
#  */
def build_training_arguments(config: DictConfig, run_dir: str | Path) -> TrainingArguments:
    training_config = config.training
    optimizer_config = config.optimizer
    adam_betas = list(optimizer_config.betas)
    if len(adam_betas) != 2:
        raise ValueError("AdamW optimizer config must define exactly two beta values.")
    output_dir = Path(run_dir) / "trainer_output"

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config.num_train_epochs),
        learning_rate=float(optimizer_config.lr),
        weight_decay=float(optimizer_config.weight_decay),
        adam_beta1=float(adam_betas[0]),
        adam_beta2=float(adam_betas[1]),
        adam_epsilon=float(optimizer_config.eps),
        lr_scheduler_type=str(training_config.lr_scheduler_type),
        warmup_ratio=float(training_config.warmup_ratio),
        max_grad_norm=float(training_config.max_grad_norm),
        seed=int(training_config.seed),
        data_seed=int(training_config.data_seed),
        per_device_train_batch_size=int(training_config.per_device_train_batch_size),
        per_device_eval_batch_size=int(training_config.per_device_eval_batch_size),
        gradient_accumulation_steps=int(training_config.gradient_accumulation_steps),
        max_steps=int(training_config.max_steps) if training_config.max_steps is not None else -1,
        logging_strategy=str(training_config.logging_strategy),
        logging_steps=int(training_config.logging_steps),
        eval_strategy=str(training_config.eval_strategy),
        eval_steps=int(training_config.eval_steps),
        save_strategy=str(training_config.save_strategy),
        save_steps=int(training_config.save_steps),
        save_total_limit=int(training_config.save_total_limit),
        load_best_model_at_end=bool(training_config.load_best_model_at_end),
        metric_for_best_model=str(training_config.metric_for_best_model),
        greater_is_better=bool(training_config.greater_is_better),
        gradient_checkpointing=bool(training_config.use_gradient_checkpointing),
        bf16=bool(training_config.bf16),
        fp16=bool(training_config.fp16),
        dataloader_num_workers=int(training_config.dataloader_num_workers),
        remove_unused_columns=bool(training_config.remove_unused_columns),
        report_to=list(training_config.report_to),
    )


# /**
#  * Создает стандартный HuggingFace Trainer для AdamW path.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @param model Загруженная causal language model.
#  * @param tokenizer Загруженный tokenizer.
#  * @param dataset Final DatasetDict с train и validation split-ами.
#  * @param run_dir Директория конкретного запуска.
#  * @return HuggingFace Trainer со стандартным optimizer path.
#  */
def build_standard_trainer(
    config: DictConfig,
    model: Any,
    tokenizer: Any,
    dataset: DatasetDict,
    run_dir: str | Path,
) -> Trainer:
    training_args = build_training_arguments(config=config, run_dir=run_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )


# /**
#  * Создает Muon trainer, когда ручная интеграция Muon будет добавлена.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @param model Загруженная causal language model.
#  * @param tokenizer Загруженный tokenizer.
#  * @param dataset Final DatasetDict с train и validation split-ами.
#  * @param run_dir Директория конкретного запуска.
#  * @return HuggingFace Trainer с Muon optimizer path.
#  */
def build_muon_trainer(
    config: DictConfig,
    model: Any,
    tokenizer: Any,
    dataset: DatasetDict,
    run_dir: str | Path,
) -> Trainer:
    raise NotImplementedError("Muon trainer is pending manual integration.")


# /**
#  * Создает combined AdamW/Muon trainer, когда этот режим будет добавлен.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @param model Загруженная causal language model.
#  * @param tokenizer Загруженный tokenizer.
#  * @param dataset Final DatasetDict с train и validation split-ами.
#  * @param run_dir Директория конкретного запуска.
#  * @return HuggingFace Trainer с combined optimizer path.
#  */
def build_combined_trainer(
    config: DictConfig,
    model: Any,
    tokenizer: Any,
    dataset: DatasetDict,
    run_dir: str | Path,
) -> Trainer:
    raise NotImplementedError("Combined trainer is pending manual integration.")


# /**
#  * Выбирает trainer path по единственному пользовательскому переключателю optimizer.name.
#  *
#  * @param config Полный Hydra-конфиг запуска.
#  * @param model Загруженная causal language model.
#  * @param tokenizer Загруженный tokenizer.
#  * @param dataset Final DatasetDict с train и validation split-ами.
#  * @param run_dir Директория конкретного запуска.
#  * @return Trainer, соответствующий выбранному optimizer mode.
#  */
def build_trainer(
    config: DictConfig,
    model: Any,
    tokenizer: Any,
    dataset: DatasetDict,
    run_dir: str | Path,
) -> Trainer:
    optimizer_name = str(config.optimizer.name).lower()
    if optimizer_name == "adamw":
        return build_standard_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            run_dir=run_dir,
        )
    if optimizer_name == "muon":
        return build_muon_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            run_dir=run_dir,
        )
    if optimizer_name == "combined":
        return build_combined_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            run_dir=run_dir,
        )
    raise NotImplementedError(f"Unsupported optimizer for training: {optimizer_name}")

# /**
#  * Копирует checkpoint directory в стабильную best/last директорию run-а.
#  *
#  * @param source_dir Исходная директория checkpoint-а.
#  * @param target_dir Целевая стабильная директория.
#  * @return Путь к целевой директории.
#  */
def copy_checkpoint_dir(source_dir: str | Path, target_dir: str | Path) -> Path:
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)
    return target_path


# /**
#  * Сохраняет best и last checkpoints в стабильные директории run-а.
#  *
#  * @param trainer HuggingFace Trainer после завершения train().
#  * @param run_dir Директория конкретного запуска.
#  * @param tokenizer Загруженный tokenizer для fallback-сохранения.
#  * @param config Полный Hydra-конфиг запуска.
#  * @return Словарь с путями checkpoints root, best и last.
#  */
def save_best_and_last_checkpoints(
    trainer: Trainer,
    run_dir: str | Path,
    tokenizer: Any,
    config: DictConfig,
) -> dict[str, str]:
    checkpoints_dir = Path(run_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_dir = checkpoints_dir / str(config.training.checkpoints.best_dir_name)
    last_dir = checkpoints_dir / str(config.training.checkpoints.last_dir_name)

    output_dir = Path(str(trainer.args.output_dir))
    last_checkpoint = find_last_checkpoint(output_dir)
    best_checkpoint = get_best_checkpoint(trainer)

    if last_checkpoint is not None:
        copy_checkpoint_dir(source_dir=last_checkpoint, target_dir=last_dir)
    else:
        trainer.save_model(str(last_dir))
        tokenizer.save_pretrained(str(last_dir))

    if best_checkpoint is not None and Path(best_checkpoint).exists():
        copy_checkpoint_dir(source_dir=best_checkpoint, target_dir=best_dir)
    else:
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))

    return {
        "local_path": str(checkpoints_dir),
        "best_path": str(best_dir),
        "last_path": str(last_dir),
    }


# /**
#  * Сохраняет final model и tokenizer в стабильные директории run-а.
#  *
#  * @param trainer HuggingFace Trainer после завершения train().
#  * @param tokenizer Загруженный tokenizer.
#  * @param run_dir Директория конкретного запуска.
#  * @return Словарь с путями model и tokenizer.
#  */
def save_final_model_artifacts(
    trainer: Trainer,
    tokenizer: Any,
    run_dir: str | Path,
) -> dict[str, str]:
    model_dir = Path(run_dir) / "model"
    tokenizer_dir = Path(run_dir) / "tokenizer"

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    return {
        "model_path": str(model_dir),
        "tokenizer_path": str(tokenizer_dir),
    }


# /**
#  * Запускает настоящий training loop.
#  *
#  * @param config Полная конфигурация обучения.
#  * @param run_dir Директория конкретного запуска.
#  * @return Training-result в общем формате с метриками и путями к артефактам.
#  */
def run_training(config: DictConfig, run_dir: str | Path) -> TrainingResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real training runs.")
    set_seed(int(config.training.seed))
    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    dataset = get_final_training_dataset(config)
    tokenizer = build_tokenizer(config.model)
    model = build_model(config.model)
    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        run_dir=run_dir,
    )

    resume_from_run_dir = config.training.get("resume_from_run_dir", None)
    resume_checkpoint = (
        find_resume_checkpoint(run_dir)
        if resume_from_run_dir is not None
        else None
    )
    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    training_time_seconds = time.perf_counter() - start_time
    completed_steps = getattr(trainer.state, "global_step", None)
    if not isinstance(completed_steps, int) or completed_steps <= 0:
        completed_steps = max(
            (
                int(entry["step"])
                for entry in trainer.state.log_history
                if isinstance(entry, dict) and isinstance(entry.get("step"), int)
            ),
            default=0,
        )
    time_per_step_seconds = (
        training_time_seconds / completed_steps if completed_steps > 0 else None
    )

    checkpoint_paths = save_best_and_last_checkpoints(
        trainer=trainer,
        run_dir=run_dir,
        tokenizer=tokenizer,
        config=config,
    )
    final_paths = save_final_model_artifacts(
        trainer=trainer,
        tokenizer=tokenizer,
        run_dir=run_dir,
    )

    result = build_training_result(
        run_name=str(config.experiment.name),
        status="completed",
        final_loss=(
            None
            if getattr(train_output, "training_loss", None) is None
            else float(train_output.training_loss)
        ),
        training_time_seconds=training_time_seconds,
        time_per_step_seconds=time_per_step_seconds,
        max_memory_mb=float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    )

    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    model_artifacts = artifacts["model"]
    tokenizer_artifacts = artifacts["tokenizer"]
    checkpoint_artifacts = artifacts["checkpoints"]
    if not isinstance(model_artifacts, dict):
        raise TypeError("Training result model artifacts must be a dictionary.")
    if not isinstance(tokenizer_artifacts, dict):
        raise TypeError("Training result tokenizer artifacts must be a dictionary.")
    if not isinstance(checkpoint_artifacts, dict):
        raise TypeError("Training result checkpoint artifacts must be a dictionary.")

    result["history"] = list(trainer.state.log_history)
    model_artifacts["local_path"] = final_paths["model_path"]
    tokenizer_artifacts["local_path"] = final_paths["tokenizer_path"]
    checkpoint_artifacts.update(checkpoint_paths)
    return result
