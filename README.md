# Optimizer Comparison

## Installation

Recommended environment:

- Python 3.10+
- CUDA-capable GPU for real training and evaluation
- Linux shell for the provided bash scripts

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and dependencies:

```bash
pip install -U pip
pip install -e .
```

Check CUDA availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

For Hugging Face Hub uploads, set a token before running scripts with `--hf-repo-id`:

```bash
export HF_TOKEN=<your_hugging_face_token>
```

For development checks:

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest
```

## Main Scripts

Train a model:

```bash
bash scripts/main/train.sh --optimizer adamw
```

Run evaluation for an existing local run:

```bash
bash scripts/main/eval.sh outputs/runs/<run_id>
```

Train and then evaluate on the same machine:

```bash
bash scripts/main/train_eval.sh --optimizer adamw
```

Useful optional arguments:

```bash
bash scripts/main/train.sh --optimizer adamw --experiment adamw_baseline
bash scripts/main/train.sh --optimizer adamw --hf-repo-id <hf_user_or_org>/<repo>
bash scripts/main/train.sh --optimizer adamw --training full --experiment adamw_baseline
```

Open MLflow UI:

```bash
mlflow ui --backend-store-uri outputs/mlruns
```

If port `5000` is busy:

```bash
mlflow ui --backend-store-uri outputs/mlruns --port 5001
```

## Workflow Scripts

Additional workflow scripts are kept in `scripts/workflows`.

Smoke train+eval:

```bash
bash scripts/workflows/smoke.sh
```

Remote train with Hugging Face Hub upload:

```bash
bash scripts/workflows/remote_train_upload.sh --optimizer adamw --hf-repo-id <hf_user_or_org>/<repo>
```

Manual training grid:

```bash
bash scripts/workflows/train_grid.sh
```

Manual evaluation grid:

```bash
bash scripts/workflows/eval_grid.sh
```
