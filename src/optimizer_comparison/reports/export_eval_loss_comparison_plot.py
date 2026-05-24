"""Export eval loss comparison plots from MLflow runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from optimizer_comparison.artifacts.local_store import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for eval loss comparison plot export."""
    parser = argparse.ArgumentParser(description="Export an eval loss comparison plot.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("RUN_ID", "LABEL"),
        required=True,
        help="MLflow run id and the label to show in the legend. Can be repeated.",
    )
    parser.add_argument("--tracking-uri", default="outputs/mlruns")
    parser.add_argument("--metric-name", default="eval/loss")
    parser.add_argument("--output-path", default="outputs/reports/eval_loss_comparison.png")
    parser.add_argument("--title", default="Eval loss comparison")
    parser.add_argument(
        "--prepend-zero-run-id",
        default=None,
        help="MLflow run id containing the baseline metric to prepend at step 0.",
    )
    parser.add_argument(
        "--prepend-zero-result",
        default=None,
        help="Path to a result JSON containing mlflow_run_id for the step-zero baseline.",
    )
    parser.add_argument(
        "--prepend-zero-value",
        type=float,
        default=None,
        help="Baseline metric value to prepend at step 0.",
    )
    return parser


def load_mlflow_run_id_from_result(path: Path) -> str:
    """Loads mlflow_run_id from a saved result JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Result JSON must contain an object: {path}")

    mlflow_run_id = data.get("mlflow_run_id", None)
    if mlflow_run_id is None:
        raise ValueError(f"Result JSON does not contain mlflow_run_id: {path}")
    return str(mlflow_run_id)


def setup_tracking(tracking_uri: str) -> None:
    """Configures MLflow tracking URI for a local or external store."""
    if "://" in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return

    mlflow.set_tracking_uri(resolve_project_path(tracking_uri).as_uri())


def get_metric_series(
    client: MlflowClient,
    run_id: str,
    metric_name: str,
) -> list[tuple[int, float]]:
    """Loads one metric history from MLflow and returns points ordered by step."""
    history = client.get_metric_history(run_id=run_id, key=metric_name)
    points = [
        (int(metric.step), float(metric.value))
        for metric in history
        if isinstance(metric.step, int)
    ]
    points.sort(key=lambda point: point[0])
    if not points:
        raise ValueError(f"Run {run_id} does not contain metric '{metric_name}'.")
    return points


def get_step_zero_metric_value(
    client: MlflowClient,
    run_id: str,
    metric_name: str,
) -> float:
    """Loads a baseline metric value from an MLflow run for step-zero plotting."""
    points = get_metric_series(client=client, run_id=run_id, metric_name=metric_name)
    step_zero_values = [value for step, value in points if step == 0]
    if step_zero_values:
        return step_zero_values[-1]
    if len(points) == 1:
        return points[0][1]
    raise ValueError(
        f"Run {run_id} has multiple '{metric_name}' points but no explicit step 0 value."
    )


def prepend_zero_point(
    series: list[tuple[int, float]],
    zero_value: float | None,
) -> list[tuple[int, float]]:
    """Prepends a shared step-zero baseline point when requested."""
    if zero_value is None:
        return series
    if series and series[0][0] == 0:
        return series
    return [(0, float(zero_value)), *series]


def save_eval_loss_comparison_plot(
    series_by_label: dict[str, list[tuple[int, float]]],
    output_path: Path,
    metric_name: str,
    title: str,
) -> Path:
    """Saves one line plot with all selected eval loss series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for label, points in series_by_label.items():
        steps, values = zip(*points, strict=False)
        axis.plot(steps, values, label=label, linewidth=1.9, marker="o", markersize=3)

    axis.set_title(title, fontsize=12)
    axis.set_xlabel("step")
    axis.set_ylabel(metric_name)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def export_eval_loss_comparison_plot(
    runs: list[tuple[str, str]],
    tracking_uri: str,
    metric_name: str,
    output_path: Path,
    title: str,
    prepend_zero_run_id: str | None,
    prepend_zero_result: Path | None,
    prepend_zero_value: float | None,
) -> Path:
    """Loads selected MLflow metric histories and saves a comparison plot."""
    configured_zero_sources = [
        source is not None
        for source in [prepend_zero_run_id, prepend_zero_result, prepend_zero_value]
    ]
    if sum(configured_zero_sources) > 1:
        raise ValueError(
            "--prepend-zero-run-id, --prepend-zero-result, and --prepend-zero-value "
            "cannot be combined."
        )

    setup_tracking(tracking_uri)
    client = MlflowClient()
    zero_value = prepend_zero_value
    if prepend_zero_result is not None:
        prepend_zero_run_id = load_mlflow_run_id_from_result(prepend_zero_result)
    if prepend_zero_run_id is not None:
        zero_value = get_step_zero_metric_value(
            client=client,
            run_id=prepend_zero_run_id,
            metric_name=metric_name,
        )

    series_by_label = {
        label: prepend_zero_point(
            series=get_metric_series(client=client, run_id=run_id, metric_name=metric_name),
            zero_value=zero_value,
        )
        for run_id, label in runs
    }
    return save_eval_loss_comparison_plot(
        series_by_label=series_by_label,
        output_path=output_path,
        metric_name=metric_name,
        title=title,
    )


def main(argv: list[str] | None = None) -> None:
    """Runs the eval loss comparison plot export CLI."""
    args = build_parser().parse_args(argv)
    output_path = export_eval_loss_comparison_plot(
        runs=[(str(run_id), str(label)) for run_id, label in args.run],
        tracking_uri=str(args.tracking_uri),
        metric_name=str(args.metric_name),
        output_path=Path(args.output_path),
        title=str(args.title),
        prepend_zero_run_id=None
        if args.prepend_zero_run_id is None
        else str(args.prepend_zero_run_id),
        prepend_zero_result=None
        if args.prepend_zero_result is None
        else Path(args.prepend_zero_result),
        prepend_zero_value=args.prepend_zero_value,
    )
    print(f"Eval loss comparison plot written to: {output_path}")


if __name__ == "__main__":
    main()
