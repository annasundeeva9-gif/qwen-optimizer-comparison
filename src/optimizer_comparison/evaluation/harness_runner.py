"""lm-evaluation-harness runner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig

from optimizer_comparison.artifacts.local_store import resolve_project_path
from optimizer_comparison.evaluation.model_source import EvaluationModelSource


def get_harness_config(config: DictConfig) -> DictConfig:
    """Returns evaluation.harness from a full or evaluation-only config."""
    evaluation_config = config.get("evaluation", None)
    if isinstance(evaluation_config, DictConfig):
        harness_config = evaluation_config.get("harness", None)
    else:
        harness_config = config.get("harness", None)

    if not isinstance(harness_config, DictConfig):
        raise ValueError("evaluation.harness config section is required.")
    return harness_config


def get_eval_samples_limit(config: DictConfig) -> object | None:
    """Returns the smoke evaluation sample limit when configured."""
    mode_config = config.get("mode", None)
    if not isinstance(mode_config, DictConfig):
        return None
    return cast(object | None, mode_config.get("limit_eval_samples", None))


def get_evaluation_artifact_dir(model_source: EvaluationModelSource) -> Path | None:
    """Returns the evaluation artifact directory for the model source."""
    if model_source.run_dir is None:
        return None
    return model_source.run_dir / "evaluation"


def get_lm_eval_output_path(config: DictConfig, model_source: EvaluationModelSource) -> Path:
    """Returns the stable lm-evaluation-harness JSON output path."""
    artifact_dir = get_evaluation_artifact_dir(model_source)
    if artifact_dir is not None:
        return artifact_dir / Path(str(get_harness_config(config).output_path)).name
    return resolve_project_path(str(get_harness_config(config).output_path))


def get_lm_eval_raw_log_path(config: DictConfig, model_source: EvaluationModelSource) -> Path:
    """Returns the stable lm-evaluation-harness raw log path."""
    artifact_dir = get_evaluation_artifact_dir(model_source)
    if artifact_dir is not None:
        return artifact_dir / Path(str(get_harness_config(config).raw_log_path)).name
    return resolve_project_path(str(get_harness_config(config).raw_log_path))


def build_lm_eval_command(
    config: DictConfig,
    model_source: EvaluationModelSource,
) -> list[str]:
    """Builds the lm-evaluation-harness command."""
    harness_config = get_harness_config(config)
    tasks = [str(task) for task in harness_config.get("tasks", [])]
    if not tasks:
        raise ValueError("evaluation.harness.tasks must contain at least one task.")

    output_path = get_lm_eval_output_path(config=config, model_source=model_source)
    command = [
        "lm-eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_source.model_path}",
        f"tokenizer={model_source.tokenizer_path}",
        "--tasks",
        *tasks,
        "--batch_size",
        str(harness_config.batch_size),
        "--device",
        "cuda",
        "--output_path",
        str(output_path),
    ]

    limit = get_eval_samples_limit(config)
    if limit is not None:
        command.extend(["--limit", str(limit)])

    return command


def write_raw_lm_eval_log(raw_log_path: str | Path, stdout: str, stderr: str) -> None:
    """Writes harness stdout and stderr into a raw log file."""
    path = resolve_project_path(raw_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"===== STDOUT =====\n{stdout}\n===== STDERR =====\n{stderr}",
        encoding="utf-8",
    )


def run_process_with_live_log(
    command: list[str],
    raw_log_path: str | Path,
    env: dict[str, str],
) -> int:
    """Runs a process with live console output and raw log capture."""
    path = resolve_project_path(raw_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as log_file:
        log_file.write("===== OUTPUT =====\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        if process.stdout is None:
            raise RuntimeError("Failed to capture lm-evaluation-harness output stream.")

        while True:
            chunk = process.stdout.read(1)
            if chunk == "" and process.poll() is not None:
                break
            if chunk == "":
                continue

            # Preserve tqdm carriage returns while still keeping a raw log copy.
            sys.stdout.write(chunk)
            sys.stdout.flush()
            log_file.write(chunk)
            log_file.flush()

        return process.wait()


def normalize_lm_eval_output_path(output_path: str | Path) -> Path:
    """Renames timestamped harness JSON output to the stable path."""
    stable_path = resolve_project_path(output_path)

    candidates = sorted(
        stable_path.parent.glob(f"{stable_path.stem}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        candidates[0].replace(stable_path)
        return stable_path

    if stable_path.is_file():
        return stable_path

    raise FileNotFoundError(f"lm-evaluation-harness result file was not found: {stable_path}")


def run_lm_eval_harness(
    config: DictConfig,
    model_source: EvaluationModelSource,
) -> dict[str, str]:
    """Runs lm-evaluation-harness and returns output paths."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for lm-evaluation-harness runs.")

    output_path = get_lm_eval_output_path(config=config, model_source=model_source)
    raw_log_path = get_lm_eval_raw_log_path(config=config, model_source=model_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_lm_eval_command(config=config, model_source=model_source)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "0"
    print("Running lm-evaluation-harness...", file=sys.stderr)
    returncode = run_process_with_live_log(
        command=command,
        raw_log_path=raw_log_path,
        env=env,
    )
    if returncode != 0:
        raise RuntimeError(
            "lm-evaluation-harness failed "
            f"with exit code {returncode}. Raw log: {raw_log_path}"
        )

    output_path = normalize_lm_eval_output_path(output_path)

    return {
        "output_path": str(output_path),
        "raw_log_path": str(raw_log_path),
    }
