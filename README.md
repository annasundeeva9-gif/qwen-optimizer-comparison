# Optimizer Comparison

Project for comparing AdamW and Muon optimization strategies on Qwen2.5-0.5B fine-tuning.

The comparison is focused on:

- training time;
- maximum memory usage;
- final model quality;
- training convergence.

The evaluation tasks are `piqa`, `arc_easy`, `arc_challenge`, `winogrande`, and `hellaswag`.

## Environment

Use a Linux machine with a CUDA-capable GPU. The main workflow scripts are Bash scripts and are intended to be run from the repository root.

Create and activate a virtual environment, then install PyTorch with CUDA support:

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
```bash
ruff check src scripts
mypy src
```

## Main Train And Eval Workflow

The expected full workflow is `scripts/main/train_eval.sh`. It trains a model, evaluates the trained checkpoint with lm-evaluation-harness, and writes local artifacts and MLflow logs. 

A smoke test can be run on a small model, a subset of the dataset, and a limited number of validation examples before launching a full experiment:

```bash
bash scripts/workflows/smoke.sh --optimizer <type> --experiment <name>
```

To launch a full run:

```bash
bash scripts/main/train_eval.sh \
  --mode full \
  --model qwen_0_5b \
  --optimizer <type> \
  --experiment <name>
```
Available optimizer types: adamw, muon.

For the hybrid Muon + AdamW variant, use the Muon optimizer with a limited number of Muon-routed layers, for example:

```bash
bash scripts/main/train_eval.sh \
  --mode full \
  --model qwen_0_5b \
  --optimizer muon \
  --experiment hybrid_l12 \
  optimizer.muon_layer_count=12
```

The project also supports a two-machine workflow: training on one machine, uploading the trained checkpoint to Hugging Face Hub, and then downloading and evaluating it on another machine. This workflow is implemented in remote_train_upload.sh and load_eval.sh. These scripts, along with additional helper scripts, are described in report/docs/ADDITIONAL_SCRIPTS.md

## Training logs

Curated logs for reported runs are stored in `report/artifacts/logs`.
Each run directory contains the resolved config and evaluation artifacts.
Per-step training metrics are produced by the training pipeline and logged to MLflow / local run artifacts.

## Report

This repository includes a short summary of the experimental results below. The full report is available as a LaTeX document and as a compiled PDF: 

 - report/docs/main_report.tex
 - report/docs/main_report.pdf

In this project, AdamW, Muon, and a hybrid Muon + AdamW strategy were compared for full fine-tuning of Qwen2.5-0.5B. The experiments track training loss, learning rate, elapsed time, and resource usage, and final model quality is evaluated with lm-evaluation-harness. The results show that Muon is sensitive to the learning rate and can perform close to AdamW when tuned carefully, while an overly large learning rate noticeably hurts convergence. The hybrid strategy shows intermediate behavior and can be considered a practical compromise, but the current experiments are limited by the number of runs, the fixed seed, and the size of the hyperparameter search.