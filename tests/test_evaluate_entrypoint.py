import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from optimizer_comparison import evaluate as evaluate_module
from optimizer_comparison.evaluate import (
    build_evaluation_result,
    get_evaluation_result_path,
    load_training_result_for_evaluation,
)
from optimizer_comparison.evaluation.model_source import EvaluationModelSource


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
