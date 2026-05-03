from pathlib import Path


# /**
#  * Возвращает корень проекта для чтения shell-скриптов.
#  *
#  * @return Путь к корню репозитория.
#  */
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# /**
#  * Проверяет основной train-скрипт с параметром optimizer.
#  *
#  * @return None.
#  */
def test_main_train_script_accepts_optimizer_argument() -> None:
    script = (project_root() / "scripts" / "main" / "train.sh").read_text(encoding="utf-8")

    assert "python -m optimizer_comparison.train" in script
    assert "--optimizer=*|--optim=*" in script
    assert "--training=*" in script
    assert '"training=${TRAINING}"' in script
    assert 'OPTIMIZER="$(echo "${OPTIMIZER}" | tr' in script
    assert '"optimizer=${OPTIMIZER}"' in script
    assert 'EXPERIMENT_CONFIG="smoke_${OPTIMIZER}_tiny"' in script
    assert 'EXPERIMENT_CONFIG="${OPTIMIZER}_baseline"' in script
    assert '"experiment=${EXPERIMENT_CONFIG}"' in script
    assert '"experiment.name=${EXPERIMENT}"' in script


# /**
#  * Проверяет, что основной train-скрипт умеет включить HF Hub upload.
#  *
#  * @return None.
#  */
def test_main_train_script_accepts_hf_repo_id() -> None:
    script = (project_root() / "scripts" / "main" / "train.sh").read_text(encoding="utf-8")

    assert "--hf-repo-id" in script
    assert "artifacts.hf_hub.use=true" in script
    assert "artifacts.hf_hub.repo_id=${HF_REPO_ID}" in script


# /**
#  * Проверяет основной eval-скрипт для локального run_dir.
#  *
#  * @return None.
#  */
def test_main_eval_script_uses_run_dir_source() -> None:
    script = (project_root() / "scripts" / "main" / "eval.sh").read_text(encoding="utf-8")

    assert "Usage: $0 <run_dir>" in script
    assert "python -m optimizer_comparison.evaluate" in script
    assert '"evaluation.source.run_dir=${RUN_DIR}"' in script


# /**
#  * Проверяет основной train+eval-скрипт.
#  *
#  * @return None.
#  */
def test_main_train_eval_script_runs_train_then_eval() -> None:
    script = (project_root() / "scripts" / "main" / "train_eval.sh").read_text(encoding="utf-8")

    assert "python -m optimizer_comparison.train" in script
    assert "--hf-repo-id" in script
    assert "artifacts.hf_hub.use=true" in script
    assert '"training=${TRAINING}"' in script
    assert 'EXPERIMENT_CONFIG="smoke_${OPTIMIZER}_tiny"' in script
    assert '"experiment=${EXPERIMENT_CONFIG}"' in script
    assert '"experiment.name=${EXPERIMENT}"' in script
    assert "RUN_DIR=\"$(ls -td " in script
    assert "python -m optimizer_comparison.evaluate" in script
    assert '"evaluation.source.run_dir=${RUN_DIR}"' in script
    assert "python scripts/workflows/upload_hf_artifact.py" in script
    assert "python scripts/workflows/upload_mlflow_snapshot.py" in script


# /**
#  * Проверяет smoke workflow поверх основного train+eval-скрипта.
#  *
#  * @return None.
#  */
def test_workflow_smoke_script_uses_tiny_qwen_model() -> None:
    script = (project_root() / "scripts" / "workflows" / "smoke.sh").read_text(
        encoding="utf-8"
    )

    assert "bash scripts/main/train_eval.sh" in script
    assert "HF_REPO_ID" in script
    assert "--hf-repo-id" in script
    assert "--mode smoke" in script
    assert "--model tiny_qwen_2_5" in script
    assert 'OPTIMIZER="adamw"' in script
    assert 'EXPERIMENT="smoke_adamw_tiny"' in script
    assert "optimizer=*)" in script
    assert "experiment.name=*)" in script
    assert '--optimizer "${OPTIMIZER}"' in script
    assert '--experiment "${EXPERIMENT}"' in script
    assert '"experiment.tags.optimizer=${OPTIMIZER}"' in script
    assert "data.final.dir=outputs/datasets/final/openwebtext_100k_smoke" in script


# /**
#  * Проверяет mock workflow.
#  *
#  * @return None.
#  */
def test_workflow_mock_script_uses_mock_config() -> None:
    script = (project_root() / "scripts" / "workflows" / "mock.sh").read_text(
        encoding="utf-8"
    )

    assert "bash scripts/main/train.sh" in script
    assert "--mode mock" in script
    assert "--model mock" in script
    assert "--optimizer adamw" in script
    assert "--experiment mock_adamw" in script


# /**
#  * Проверяет remote workflow с обязательной загрузкой artifacts и MLflow snapshot-а.
#  *
#  * @return None.
#  */
def test_workflow_remote_train_upload_requires_hf_repo() -> None:
    script = (project_root() / "scripts" / "workflows" / "remote_train_upload.sh").read_text(
        encoding="utf-8"
    )

    assert "HF repo id is required" in script
    assert "bash scripts/main/train.sh" in script
    assert "--hf-repo-id=${HF_REPO_ID}" in script
    assert "python scripts/workflows/upload_mlflow_snapshot.py" in script


# /**
#  * Проверяет train grid workflow с ручным списком запусков.
#  *
#  * @return None.
#  */
def test_workflow_train_grid_script_uploads_mlflow_snapshot() -> None:
    script = (project_root() / "scripts" / "workflows" / "train_grid.sh").read_text(
        encoding="utf-8"
    )

    assert "RUNS=(" in script
    assert "--optimizer adamw --experiment adamw_baseline" in script
    assert "--optimizer muon --experiment muon_baseline" in script
    assert "python scripts/workflows/upload_mlflow_snapshot.py" in script
    assert "HF_REPO_ID" in script


# /**
#  * Проверяет eval grid workflow с ручным списком run_dir-ов.
#  *
#  * @return None.
#  */
def test_workflow_eval_grid_script_uses_manual_run_dirs() -> None:
    script = (project_root() / "scripts" / "workflows" / "eval_grid.sh").read_text(
        encoding="utf-8"
    )

    assert "RUN_DIRS=(" in script
    assert "REPLACE_ME" in script
    assert "bash scripts/main/eval.sh" in script


# /**
#  * Проверяет workflow загрузки remote run-а, merge MLflow snapshot-а и запуска eval.
#  *
#  * @return None.
#  */
def test_workflow_load_eval_downloads_run_and_mlruns() -> None:
    script = (project_root() / "scripts" / "workflows" / "load_eval.sh").read_text(
        encoding="utf-8"
    )

    assert "--run-id" in script
    assert "python scripts/workflows/download_mlflow_snapshot.py" in script
    assert "--skip-mlflow" in script
    assert "tracking.enabled=false" in script
    assert "python -m optimizer_comparison.evaluate" in script
    assert "evaluation.source.use_hf_hub=true" in script
    assert "evaluation.source.repo_path=runs/${RUN_ID}" in script
    assert "evaluation.source.download_dir=outputs" in script
