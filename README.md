# Optimizer Comparison

Project for comparing AdamW and Muon optimization strategies on Qwen2.5-0.5B fine-tuning.

The comparison is focused on:

- training time;
- maximum memory usage;
- final model quality;
- training convergence.

The evaluation tasks are `piqa`, `arc_easy`, `arc_challenge`, `winogrande`, and `hellaswag`.

## Environment

Use a Linux machine with a CUDA-capable GPU. The main workflow scripts are Bash scripts and are intended to run from the repository root.

Create and activate a virtual environment, install PyTorch with CUDA support.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

For development checks:

```bash
pip install -e ".[dev]"
```

## Main Train And Eval Workflow

The expected full workflow is `scripts/main/train_eval.sh`. It trains a model, evaluates the trained checkpoint with lm-evaluation-harness, and writes local artifacts and MLflow logs. 

A useful smoke test on a subset of the dataset, with a small model, and on a subset of validation examples before a full run:

```bash
bash scripts/workflows/smoke.sh --optimizer <type> --experiment <name>
```

Launching the full run:

```bash
bash scripts/main/train_eval.sh \
  --mode full \
  --model qwen_0_5b \
  --optimizer <type> \
  --experiment <name>
```
Available optimizers: adam/muon. For the hybrid Muon+AdamW variant add:

```bash
optimizer.muon_layer_count=12
```

The project also supports an additional pipeline, which involves launching on one machine, saving the results via a hugging face hub, and then loading and running validation on another with scripts remote_train_upload.sh and load_eval.sh.

## Reproducing report figures and tables

All training and evaluation artifacts are stored with run_id = name_timestamp:

```text
outputs/runs/<run_id>/
```
To get the results from the report, run scripts export_comparison_table, export_task_metrics_table, export_short_comparison_table

```bash
python -m optimizer_comparison.reports.<script> \
  <run_id> \
  ... \
  <run_id> \
  --output outputs/reports/<name>
```