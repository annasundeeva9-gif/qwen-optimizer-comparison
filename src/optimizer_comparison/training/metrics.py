"""Placeholder helpers for future report metrics."""

from __future__ import annotations


def collect_training_metrics() -> dict[str, float]:
    """Placeholder for extra training metrics that are not used by the pipeline."""
    raise NotImplementedError(
        "Extra report metrics placeholder is not used by the training pipeline."
    )
