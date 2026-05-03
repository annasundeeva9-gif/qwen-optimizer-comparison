import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from optimizer_comparison import evaluate as evaluate_module
from optimizer_comparison.evaluate import (
    add_evaluation_summary_path,
    build_base_model_evaluation_result,
    build_evaluation_result,
    get_evaluation_result_path,
    load_existing_evaluation_mlflow_run_id,
    load_training_result_for_evaluation,
)
from optimizer_comparison.evaluation.model_source import (
    BASE_QWEN_MODEL_ID,
    BASE_QWEN_RUN_ID,
    BASE_QWEN_RUN_NAME,
    EvaluationModelSource,
)


# /**
#  * Создает model source с run directory для evaluate-тестов.
#  *
#  * @param run_dir Директория training run-а.
#  * @return EvaluationModelSource.
#  */
def make_model_source(run_dir: Path) -> EvaluationModelSource:
    return EvaluationModelSource(
        source_type="local",
        run_dir=run_dir,
        model_path=run_dir / "model",
        tokenizer_path=run_dir / "tokenizer",
    )


# /**
#  * Создает source фиксированной базовой модели для evaluate-тестов.
#  *
#  * @param run_dir Директория baseline evaluation run-а.
#  * @return EvaluationModelSource.
#  */
def make_base_model_source(run_dir: Path) -> EvaluationModelSource:
    return EvaluationModelSource(
        source_type="base_model",
        run_dir=run_dir,
        model_path=BASE_QWEN_MODEL_ID,
        tokenizer_path=BASE_QWEN_MODEL_ID,
    )


# /**
#  * Создает минимальный training result.json.
#  *
#  * @param run_dir Директория training run-а.
#  * @return Training result.
#  */
def write_training_result(run_dir: Path) -> dict[str, Any]:
    training_result = {
        "status": "completed",
        "run_id": "adamw_baseline__2026-05-01_10-00-00",
        "run_name": "adamw_baseline",
        "mlflow_run_id": "mlflow-run-id",
    }
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps(training_result), encoding="utf-8")
    return training_result


# /**
#  * Проверяет чтение training result из run directory.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_training_result_for_evaluation_reads_result_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "adamw_baseline__2026-05-01_10-00-00"
    expected_result = write_training_result(run_dir)

    result = load_training_result_for_evaluation(make_model_source(run_dir))

    assert result == expected_result


# /**
#  * Проверяет понятную ошибку, если training result не содержит mlflow_run_id.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_evaluation_result_requires_mlflow_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="mlflow_run_id"):
        build_evaluation_result(
            training_result={"run_id": "run"},
            model_source=make_model_source(run_dir),
            harness_paths={
                "output_path": str(run_dir / "evaluation" / "lm_eval_results.json"),
                "raw_log_path": str(run_dir / "evaluation" / "lm_eval_stdout.txt"),
                "summary_path": str(run_dir / "evaluation" / "evaluation_summary.csv"),
            },
        )


# /**
#  * Проверяет сборку project-level evaluation result.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_evaluation_result_uses_training_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    model_source = make_model_source(run_dir)

    evaluation_result = build_evaluation_result(
        training_result={
            "run_id": "run",
            "run_name": "adamw_baseline",
            "mlflow_run_id": "mlflow-run-id",
        },
        model_source=model_source,
        harness_paths={
            "output_path": str(run_dir / "evaluation" / "lm_eval_results.json"),
            "raw_log_path": str(run_dir / "evaluation" / "lm_eval_stdout.txt"),
            "summary_path": str(run_dir / "evaluation" / "evaluation_summary.csv"),
        },
    )

    assert evaluation_result["status"] == "completed"
    assert evaluation_result["run_id"] == "run"
    assert evaluation_result["run_name"] == "adamw_baseline"
    assert evaluation_result["mlflow_run_id"] == "mlflow-run-id"
    assert evaluation_result["model_source"]["run_dir"] == str(run_dir)
    assert evaluation_result["lm_eval_result_path"] == str(
        run_dir / "evaluation" / "lm_eval_results.json"
    )
    assert evaluation_result["summary_path"] == str(
        run_dir / "evaluation" / "evaluation_summary.csv"
    )


# /**
#  * Проверяет сборку result для evaluation фиксированной базовой модели.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_base_model_evaluation_result_uses_fixed_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / BASE_QWEN_RUN_ID

    evaluation_result = build_base_model_evaluation_result(
        model_source=make_base_model_source(run_dir),
        harness_paths={
            "output_path": str(run_dir / "evaluation" / "lm_eval_results.json"),
            "raw_log_path": str(run_dir / "evaluation" / "lm_eval_stdout.txt"),
            "summary_path": str(run_dir / "evaluation" / "evaluation_summary.csv"),
        },
    )

    assert evaluation_result["run_id"] == BASE_QWEN_RUN_ID
    assert evaluation_result["run_name"] == BASE_QWEN_RUN_NAME
    assert evaluation_result["mlflow_run_id"] is None
    assert evaluation_result["model_source"]["type"] == "base_model"
    assert evaluation_result["model_source"]["base_model_id"] == BASE_QWEN_MODEL_ID
    assert evaluation_result["lm_eval_result_path"] == str(
        run_dir / "evaluation" / "lm_eval_results.json"
    )


# /**
#  * Проверяет добавление CSV summary path к результатам lm-evaluation-harness.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_add_evaluation_summary_path_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "evaluation" / "lm_eval_results.json"
    output_path.parent.mkdir()
    output_path.write_text(
        json.dumps({"results": {"piqa": {"acc,none": 0.75}}}),
        encoding="utf-8",
    )

    harness_paths = add_evaluation_summary_path(
        {
            "output_path": str(output_path),
            "raw_log_path": str(tmp_path / "evaluation" / "lm_eval_stdout.txt"),
        }
    )

    assert harness_paths["summary_path"] == str(tmp_path / "evaluation" / "evaluation_summary.csv")
    assert (tmp_path / "evaluation" / "evaluation_summary.csv").is_file()


# /**
#  * Проверяет путь project-level evaluation result.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_get_evaluation_result_path_uses_run_evaluation_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = OmegaConf.create(
        {"evaluation": {"harness": {"result_filename": "evaluation_result.json"}}}
    )

    result_path = get_evaluation_result_path(
        config=config,
        model_source=make_model_source(run_dir),
    )

    assert result_path == run_dir / "evaluation" / "evaluation_result.json"


# /**
#  * Проверяет чтение MLflow run id из предыдущего evaluation result.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_load_existing_evaluation_mlflow_run_id_reads_previous_result(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "evaluation_result.json"
    result_path.write_text(
        json.dumps({"mlflow_run_id": "existing-mlflow-run-id"}),
        encoding="utf-8",
    )

    mlflow_run_id = load_existing_evaluation_mlflow_run_id(result_path)

    assert mlflow_run_id == "existing-mlflow-run-id"


# /**
#  * Проверяет orchestration main без запуска реального harness-а.
#  *
#  * @param monkeypatch Инструмент pytest для подмены шагов evaluation.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_main_orchestrates_evaluation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_training_result(run_dir)
    model_source = make_model_source(run_dir)
    logged_results: list[dict[str, Any]] = []

    monkeypatch.setattr(
        evaluate_module,
        "resolve_evaluation_model_source",
        lambda config: model_source,
    )
    monkeypatch.setattr(
        evaluate_module,
        "run_lm_eval_harness",
        lambda config, model_source: {
            "output_path": str(run_dir / "evaluation" / "lm_eval_results.json"),
            "raw_log_path": str(run_dir / "evaluation" / "lm_eval_stdout.txt"),
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "add_evaluation_summary_path",
        lambda harness_paths: {
            **harness_paths,
            "summary_path": str(run_dir / "evaluation" / "evaluation_summary.csv"),
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "log_evaluation_run",
        lambda config, evaluation_result: logged_results.append(evaluation_result),
    )

    evaluate_module.main.__wrapped__(
        OmegaConf.create(
            {"evaluation": {"harness": {"result_filename": "evaluation_result.json"}}}
        )
    )

    result_path = run_dir / "evaluation" / "evaluation_result.json"
    assert result_path.is_file()
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_result["run_id"] == "adamw_baseline__2026-05-01_10-00-00"
    assert logged_results == [saved_result]


# /**
#  * Проверяет orchestration для baseline evaluation без training result.json.
#  *
#  * @param monkeypatch Инструмент pytest для подмены шагов evaluation.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_main_orchestrates_base_model_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / BASE_QWEN_RUN_ID
    model_source = make_base_model_source(run_dir)

    monkeypatch.setattr(
        evaluate_module,
        "resolve_evaluation_model_source",
        lambda config: model_source,
    )
    monkeypatch.setattr(
        evaluate_module,
        "run_lm_eval_harness",
        lambda config, model_source: {
            "output_path": str(run_dir / "evaluation" / "lm_eval_results.json"),
            "raw_log_path": str(run_dir / "evaluation" / "lm_eval_stdout.txt"),
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "add_evaluation_summary_path",
        lambda harness_paths: {
            **harness_paths,
            "summary_path": str(run_dir / "evaluation" / "evaluation_summary.csv"),
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "log_evaluation_run",
        lambda config, evaluation_result: "base-mlflow-run-id",
    )

    evaluate_module.main.__wrapped__(
        OmegaConf.create(
            {"evaluation": {"harness": {"result_filename": "evaluation_result.json"}}}
        )
    )

    result_path = run_dir / "evaluation" / "evaluation_result.json"
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_result["run_id"] == BASE_QWEN_RUN_ID
    assert saved_result["mlflow_run_id"] == "base-mlflow-run-id"
    assert saved_result["model_source"]["base_model_id"] == BASE_QWEN_MODEL_ID


# /**
#  * Проверяет, что повторный baseline evaluation переиспользует старый MLflow run id.
#  *
#  * @param monkeypatch Инструмент pytest для подмены шагов evaluation.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_main_reuses_base_model_mlflow_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / BASE_QWEN_RUN_ID
    result_path = run_dir / "evaluation" / "evaluation_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"mlflow_run_id": "existing-base-mlflow-run-id"}),
        encoding="utf-8",
    )
    model_source = make_base_model_source(run_dir)
    logged_results: list[dict[str, Any]] = []

    monkeypatch.setattr(
        evaluate_module,
        "resolve_evaluation_model_source",
        lambda config: model_source,
    )
    monkeypatch.setattr(
        evaluate_module,
        "run_lm_eval_harness",
        lambda config, model_source: {
            "output_path": str(run_dir / "evaluation" / "lm_eval_results.json"),
            "raw_log_path": str(run_dir / "evaluation" / "lm_eval_stdout.txt"),
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "add_evaluation_summary_path",
        lambda harness_paths: {
            **harness_paths,
            "summary_path": str(run_dir / "evaluation" / "evaluation_summary.csv"),
        },
    )

    def fake_log_evaluation_run(config: Any, evaluation_result: dict[str, Any]) -> str:
        logged_results.append(evaluation_result)
        return str(evaluation_result["mlflow_run_id"])

    monkeypatch.setattr(
        evaluate_module,
        "log_evaluation_run",
        fake_log_evaluation_run,
    )

    evaluate_module.main.__wrapped__(
        OmegaConf.create(
            {"evaluation": {"harness": {"result_filename": "evaluation_result.json"}}}
        )
    )

    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert logged_results[0]["mlflow_run_id"] == "existing-base-mlflow-run-id"
    assert saved_result["mlflow_run_id"] == "existing-base-mlflow-run-id"
