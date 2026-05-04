# Additional Scripts

- `scripts/main/train_eval.sh` - main one-machine train and eval pipeline.
- `scripts/workflows/train.sh` - train-only helper.
- `scripts/workflows/eval.sh` - eval-only helper for an existing local run.
- `scripts/workflows/smoke.sh` - small train+eval smoke run.
- `scripts/workflows/mock.sh` - mocked train run without real training.
- `scripts/workflows/remote_train_upload.sh` - train and upload artifacts to Hugging Face Hub.
- `scripts/workflows/load_eval.sh` - download a Hub run and evaluate it locally.
- `scripts/workflows/train_grid.sh` - manual train grid launcher.
- `scripts/workflows/eval_grid.sh` - manual eval grid launcher.
- `scripts/workflows/upload_hf_artifact.py` - upload one local artifact to Hugging Face Hub.
- `scripts/workflows/upload_mlflow_snapshot.py` - upload an MLflow snapshot to Hugging Face Hub.
- `scripts/workflows/download_mlflow_snapshot.py` - download and merge an MLflow snapshot.
- `scripts/workflows/log_existing_eval.py` - log an already saved evaluation result to MLflow.
- `python -m optimizer_comparison.reports.export_run_logs` - copy selected run logs for report artifacts.
