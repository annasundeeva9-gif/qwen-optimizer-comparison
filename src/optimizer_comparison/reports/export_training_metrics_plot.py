"""Export training loss and grad norm plots for a selected run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeGuard

from omegaconf import OmegaConf


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for training metric plot export."""
    parser = argparse.ArgumentParser(description="Export training loss and grad norm plots.")
    parser.add_argument("run_id")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--loss-output", default=None)
    parser.add_argument("--grad-norm-output", default=None)
    parser.add_argument("--label", default=None)
    return parser


def is_plot_number(value: object) -> TypeGuard[int | float]:
    """Checks whether a value can be plotted as a numeric point."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def load_json_dict(path: Path) -> dict[str, Any]:
    """Loads a JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


def collect_training_metric_series(
    history: list[object],
) -> dict[str, list[tuple[int, float]]]:
    """Collects train loss, eval loss, and grad norm series from history."""
    series: dict[str, list[tuple[int, float]]] = {
        "train/loss": [],
        "eval/loss": [],
        "train/grad_norm": [],
    }

    metric_map = {
        "loss": "train/loss",
        "eval_loss": "eval/loss",
        "grad_norm": "train/grad_norm",
    }
    for entry in history:
        if not isinstance(entry, dict):
            continue

        step = entry.get("step", None)
        if not isinstance(step, int):
            continue

        for source_name, target_name in metric_map.items():
            value = entry.get(source_name, None)
            if is_plot_number(value):
                series[target_name].append((step, float(value)))

    return series


def has_plot_points(
    series: dict[str, list[tuple[int, float]]],
    metric_names: list[str],
) -> bool:
    """Checks whether any selected series has points to plot."""
    return any(series.get(metric_name, []) for metric_name in metric_names)


def resolve_plot_label(config: Any, custom_label: str | None, run_id: str) -> str:
    """Resolves the plot label from CLI, config, or run id."""
    if custom_label:
        return custom_label

    optimizer_name = OmegaConf.select(config, "optimizer.name")
    if isinstance(optimizer_name, str) and optimizer_name:
        return optimizer_name

    return run_id


def save_metric_plot(
    series: dict[str, list[tuple[int, float]]],
    metric_names: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
    styles: dict[str, dict[str, Any]],
    empty_error: str,
) -> Path:
    """Saves one PNG plot for selected metric series."""
    if not has_plot_points(series=series, metric_names=metric_names):
        raise ValueError(empty_error)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for metric_name in metric_names:
        points = series.get(metric_name, [])
        if not points:
            continue
        steps, values = zip(*points, strict=False)
        axis.plot(steps, values, label=metric_name, **styles[metric_name])

    axis.set_title(title, fontsize=12)
    axis.set_xlabel("step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_training_metrics_plot(
    series: dict[str, list[tuple[int, float]]],
    loss_output_path: Path,
    grad_norm_output_path: Path,
    label: str,
    lr: object,
) -> tuple[Path, Path]:
    """Saves separate PNG plots for loss curves and grad norm."""
    title = f"{label} | lr={lr}"
    loss_path = save_metric_plot(
        series=series,
        metric_names=["train/loss", "eval/loss"],
        output_path=loss_output_path,
        title=title,
        ylabel="loss",
        styles={
            "train/loss": {"color": "#1f77b4", "linewidth": 1.8},
            "eval/loss": {
                "color": "#d62728",
                "linewidth": 1.8,
                "marker": "o",
                "markersize": 3,
            },
        },
        empty_error="Run history does not contain train/loss or eval/loss.",
    )
    grad_norm_path = save_metric_plot(
        series=series,
        metric_names=["train/grad_norm"],
        output_path=grad_norm_output_path,
        title=title,
        ylabel="grad norm",
        styles={"train/grad_norm": {"color": "#9467bd", "linewidth": 1.5}},
        empty_error="Run history does not contain train/grad_norm.",
    )
    return loss_path, grad_norm_path


def export_training_metrics_plot(
    run_id: str,
    runs_dir: Path,
    output_dir: Path,
    loss_output_path: Path | None,
    grad_norm_output_path: Path | None,
    custom_label: str | None,
) -> tuple[Path, Path]:
    """Builds and saves training metric plots for one run."""
    run_dir = runs_dir / run_id
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.yaml"
    if not result_path.is_file():
        raise FileNotFoundError(f"Run result.json was not found: {result_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config.yaml was not found: {config_path}")

    result = load_json_dict(result_path)
    config = OmegaConf.load(config_path)
    history = result.get("history", [])
    if not isinstance(history, list):
        raise TypeError(f"Run history must be a list: {result_path}")

    lr = OmegaConf.select(config, "optimizer.lr", default="-")
    label = resolve_plot_label(config=config, custom_label=custom_label, run_id=run_id)
    loss_output = loss_output_path or output_dir / f"{run_id}_losses.png"
    grad_norm_output = grad_norm_output_path or output_dir / f"{run_id}_grad_norm.png"
    return save_training_metrics_plot(
        series=collect_training_metric_series(history),
        loss_output_path=loss_output,
        grad_norm_output_path=grad_norm_output,
        label=label,
        lr=lr,
    )


def main(argv: list[str] | None = None) -> None:
    """Runs the training metric plot export CLI."""
    args = build_parser().parse_args(argv)
    loss_output_path = Path(args.loss_output) if args.loss_output is not None else None
    grad_norm_output_path = (
        Path(args.grad_norm_output) if args.grad_norm_output is not None else None
    )
    loss_path, grad_norm_path = export_training_metrics_plot(
        run_id=args.run_id,
        runs_dir=Path(args.runs_dir),
        output_dir=Path(args.output_dir),
        loss_output_path=loss_output_path,
        grad_norm_output_path=grad_norm_output_path,
        custom_label=args.label,
    )
    print(f"Loss plot written to: {loss_path}")
    print(f"Grad norm plot written to: {grad_norm_path}")


if __name__ == "__main__":
    main()
