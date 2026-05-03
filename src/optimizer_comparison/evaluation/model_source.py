"""Evaluation model artifact source resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from optimizer_comparison.artifacts.hf_hub import download_artifacts_from_hf
from optimizer_comparison.artifacts.local_store import resolve_project_path

BASE_QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B"
BASE_QWEN_RUN_ID = "__base_qwen_2_5_0_5b__"
BASE_QWEN_RUN_NAME = "base_qwen_2_5_0_5b"
BASE_QWEN_RUN_DIR = "outputs/runs/__base_qwen_2_5_0_5b__"


# /**
#  * Описывает локально доступные model/tokenizer artifacts для evaluation.
#  *
#  * @param source_type Тип источника: local, hf_hub или base_model.
#  * @param run_dir Локальная директория training run-а, если она известна.
#  * @param model_path Локальный путь к модели или фиксированный HF id для base model.
#  * @param tokenizer_path Локальный путь к токенизатору или фиксированный HF id для base model.
#  * @param hf_repo_id HF repo id, если artifacts восстановлены из Hub.
#  * @param hf_artifact_path Путь artifacts внутри HF repo, если использовался Hub.
#  * @param hf_revision Revision HF repo, если он задан.
#  */
@dataclass(frozen=True)
class EvaluationModelSource:
    source_type: str
    run_dir: Path | None
    model_path: Path | str
    tokenizer_path: Path | str
    hf_repo_id: str | None = None
    hf_artifact_path: str | None = None
    hf_revision: str | None = None


# /**
#  * Возвращает evaluation.source из полного или evaluation-only конфига.
#  *
#  * @param config Полный Hydra-конфиг или только evaluation-секция.
#  * @return Конфиг source.
#  */
def get_evaluation_source_config(config: DictConfig) -> DictConfig:
    evaluation_config = config.get("evaluation", None)
    if isinstance(evaluation_config, DictConfig):
        source_config = evaluation_config.get("source", None)
    else:
        source_config = config.get("source", None)

    if not isinstance(source_config, DictConfig):
        raise ValueError("evaluation.source config section is required.")
    return source_config


# /**
#  * Проверяет, что путь указывает на существующую директорию artifact-а.
#  *
#  * @param path Путь из конфига.
#  * @param field_name Имя поля для понятного текста ошибки.
#  * @return Абсолютный путь к директории.
#  */
def resolve_existing_artifact_dir(path: str | Path | None, field_name: str) -> Path:
    if path is None:
        raise ValueError(f"{field_name} must be set for evaluation.")

    resolved_path = resolve_project_path(path)
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"{field_name} does not exist or is not a directory: {path}")
    return resolved_path

# /**
#  * Разрешает локальный source model/tokenizer artifacts.
#  *
#  * @param source_config Evaluation source-конфиг.
#  * @return Описание локально доступных artifacts.
#  */
def resolve_local_model_source(source_config: DictConfig) -> EvaluationModelSource:
    run_dir_value = source_config.get("run_dir", None)
    if run_dir_value is not None:
        run_dir = resolve_existing_artifact_dir(path=run_dir_value, 
                                                field_name="evaluation.source.run_dir")
        model_path = resolve_existing_artifact_dir(
            path=run_dir / "model",
            field_name="evaluation.source.run_dir/model",
        )
        tokenizer_path = resolve_existing_artifact_dir(
            path=run_dir / "tokenizer",
            field_name="evaluation.source.run_dir/tokenizer",
        )
    else:
        run_dir = None
        model_path = resolve_existing_artifact_dir(
            path=source_config.get("model_path", None),
            field_name="evaluation.source.model_path",
        )
        tokenizer_path = resolve_existing_artifact_dir(
            path=source_config.get("tokenizer_path", None),
            field_name="evaluation.source.tokenizer_path",
        )
        if model_path.parent == tokenizer_path.parent:
            run_dir = model_path.parent

    return EvaluationModelSource(
        source_type="local",
        run_dir=run_dir,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


# /**
#  * Скачивает artifacts из HF Hub и возвращает локальные model/tokenizer paths.
#  *
#  * @param source_config Evaluation source-конфиг.
#  * @return Описание локально доступных artifacts.
#  */
def resolve_hf_hub_model_source(source_config: DictConfig) -> EvaluationModelSource:
    repo_id = source_config.get("repo_id", None)
    repo_path = source_config.get("repo_path", None)
    download_dir = source_config.get("download_dir", None)
    if repo_id is None:
        raise ValueError("evaluation.source.repo_id must be set when use_hf_hub=true.")
    if repo_path is None:
        raise ValueError("evaluation.source.repo_path must be set when use_hf_hub=true.")
    if download_dir is None:
        raise ValueError("evaluation.source.download_dir must be set when use_hf_hub=true.")

    revision = source_config.get("revision", None)
    token_env_var = str(source_config.get("token_env_var", "HF_TOKEN"))
    restored_path = download_artifacts_from_hf(
        repo_id=str(repo_id),
        target_dir=resolve_project_path(str(download_dir)),
        repo_path=str(repo_path),
        token=os.environ.get(token_env_var),
        revision=None if revision is None else str(revision),
    )

    model_path = resolve_existing_artifact_dir(
        path=restored_path / "model",
        field_name="downloaded evaluation model path",
    )
    tokenizer_path = resolve_existing_artifact_dir(
        path=restored_path / "tokenizer",
        field_name="downloaded evaluation tokenizer path",
    )

    return EvaluationModelSource(
        source_type="hf_hub",
        run_dir=restored_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        hf_repo_id=str(repo_id),
        hf_artifact_path=str(repo_path),
        hf_revision=None if revision is None else str(revision),
    )


# /**
#  * Возвращает фиксированный источник для evaluation базовой Qwen2.5-0.5B.
#  *
#  * Это специальный baseline-случай проекта, а не общий механизм выбора HF-моделей.
#  *
#  * @return Описание source с HF id модели и заметной run directory для результатов.
#  */
def resolve_base_qwen_model_source() -> EvaluationModelSource:
    return EvaluationModelSource(
        source_type="base_model",
        run_dir=resolve_project_path(BASE_QWEN_RUN_DIR),
        model_path=BASE_QWEN_MODEL_ID,
        tokenizer_path=BASE_QWEN_MODEL_ID,
    )


# /**
#  * Выбирает local или HF Hub источник model/tokenizer artifacts для evaluation.
#  *
#  * @param config Полный Hydra-конфиг или только evaluation-секция.
#  * @return Описание локально доступных artifacts.
#  */
def resolve_evaluation_model_source(config: DictConfig) -> EvaluationModelSource:
    source_config = get_evaluation_source_config(config)
    if bool(source_config.get("use_base_model", False)):
        if bool(source_config.get("use_hf_hub", False)):
            raise ValueError("evaluation.source.use_base_model cannot be combined with use_hf_hub.")
        return resolve_base_qwen_model_source()

    if bool(source_config.get("use_hf_hub", False)):
        return resolve_hf_hub_model_source(source_config)
    return resolve_local_model_source(source_config)
