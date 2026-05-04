"""Local artifact storage helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolves a project-relative path to an absolute path."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return get_project_root() / candidate


def build_run_id(experiment_name: str, timestamp: str) -> str:
    """Builds a readable run id from experiment name and timestamp."""
    return f"{experiment_name}__{timestamp}"


def create_run_dir(
    artifacts_root_dir: str | Path,
    experiment_name: str,
    timestamp: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Creates the local directory for one run."""
    run_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_run_id = run_id or build_run_id(
        experiment_name=experiment_name,
        timestamp=run_timestamp,
    )
    run_dir = resolve_project_path(artifacts_root_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_resolved_config(
    config: DictConfig,
    run_dir: str | Path,
    filename: str = "config.yaml",
) -> Path:
    """Saves the fully resolved Hydra config into the run directory."""
    output_path = Path(run_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_path, resolve=True)
    return output_path


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    """Saves a dictionary as JSON in the local artifact store."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
