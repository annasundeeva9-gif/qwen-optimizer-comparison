"""Training plot artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard


def is_plot_number(value: object) -> TypeGuard[int | float]:
    """Checks whether a value can be plotted as a numeric point."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def collect_training_curve_series(
    history: list[object],
) -> dict[str, list[tuple[int, float]]]:
    """Collects train loss, validation loss, and learning rate series."""
    series: dict[str, list[tuple[int, float]]] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    for entry in history:
        if not isinstance(entry, dict):
            continue

        step = entry.get("step", None)
        if not isinstance(step, int):
            continue

        metric_map = {
            "loss": "train_loss",
            "eval_loss": "val_loss",
            "learning_rate": "learning_rate",
        }
        for source_name, target_name in metric_map.items():
            value = entry.get(source_name, None)
            if is_plot_number(value):
                series[target_name].append((step, float(value)))

    return series


def has_training_curve_points(series: dict[str, list[tuple[int, float]]]) -> bool:
    """Checks whether any training curve series has points."""
    return any(points for points in series.values())


def save_training_curves_plot(
    history: list[object],
    output_path: str | Path,
) -> Path | None:
    """Saves one PNG with train loss, validation loss, and learning rate."""
    series = collect_training_curve_series(history)
    if not has_training_curve_points(series):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, loss_axis = plt.subplots(figsize=(9, 5))
    lr_axis = loss_axis.twinx()

    if series["train_loss"]:
        steps, values = zip(*series["train_loss"], strict=False)
        loss_axis.plot(steps, values, label="train loss", color="#1f77b4", linewidth=1.8)
    if series["val_loss"]:
        steps, values = zip(*series["val_loss"], strict=False)
        loss_axis.plot(steps, values, label="val loss", color="#d62728", linewidth=1.8)
    if series["learning_rate"]:
        steps, values = zip(*series["learning_rate"], strict=False)
        lr_axis.plot(steps, values, label="learning rate", color="#2ca02c", linewidth=1.5)

    loss_axis.set_xlabel("step")
    loss_axis.set_ylabel("loss")
    lr_axis.set_ylabel("learning rate")
    loss_axis.grid(True, alpha=0.25)

    lines = loss_axis.get_lines() + lr_axis.get_lines()
    labels = [str(line.get_label()) for line in lines]
    loss_axis.legend(lines, labels, loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def add_training_curves_artifact(
    result: dict[str, object],
    run_dir: str | Path,
) -> dict[str, object]:
    """Adds the training curves path to result artifacts when a plot is created."""
    history = result.get("history", [])
    if not isinstance(history, list):
        raise TypeError("Training result history must be a list.")

    plot_path = save_training_curves_plot(
        history=history,
        output_path=Path(run_dir) / "training_curves.png",
    )
    if plot_path is None:
        return result

    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Training result artifacts must be a dictionary.")

    plots = artifacts.get("plots", {})
    if not isinstance(plots, dict):
        raise TypeError("Training result artifacts.plots must be a dictionary.")

    plots["training_curves_path"] = str(plot_path)
    return result
