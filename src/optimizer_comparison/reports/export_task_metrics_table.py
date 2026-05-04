"""Build task-oriented metrics CSV for selected runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

BASE_QWEN_RUN_ID = "__base_qwen_2_5_0_5b__"
PREFERRED_QUALITY_METRICS = ("acc_norm,none", "acc,none")


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for task metrics table export."""
    parser = argparse.ArgumentParser(description="Export task metrics table for selected runs.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output", default="outputs/reports/task_metrics.csv")
    parser.add_argument("--latex-output", default=None)
    return parser


def load_json_dict(path: Path) -> dict[str, Any]:
    """Loads a JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


def find_last_history_metric(history: object, metric_name: str) -> float | None:
    """Finds the last numeric metric value in Trainer history."""
    if not isinstance(history, list):
        return None

    last_value: float | None = None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        value = entry.get(metric_name, None)
        if isinstance(value, int | float) and not isinstance(value, bool):
            last_value = float(value)
    return last_value


def load_evaluation_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    """Loads rows from evaluation_summary.csv."""
    if not summary_path.is_file():
        return []

    with summary_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def calculate_avg_harness_score(summary_rows: list[dict[str, str]]) -> float | None:
    """Calculates one average quality metric across harness tasks."""
    values_by_task: dict[str, float] = {}
    rows_by_task: dict[str, list[dict[str, str]]] = {}
    for row in summary_rows:
        task = row.get("task", "")
        if task:
            rows_by_task.setdefault(task, []).append(row)

    for task, rows in rows_by_task.items():
        for metric_name in PREFERRED_QUALITY_METRICS:
            value = next(
                (
                    row.get("value", "")
                    for row in rows
                    if row.get("metric", "") == metric_name
                ),
                "",
            )
            if value:
                values_by_task[task] = float(value)
                break

    if not values_by_task:
        return None
    return sum(values_by_task.values()) / len(values_by_task)


def format_metric(value: object, digits: int = 4) -> str:
    """Formats a numeric metric for CSV output."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return "-"


def build_base_model_row(run_id: str, run_dir: Path) -> dict[str, str]:
    """Builds the task metrics row for the base model evaluation."""
    avg_harness_score = calculate_avg_harness_score(
        load_evaluation_summary_rows(run_dir / "evaluation" / "evaluation_summary.csv")
    )
    return {
        "run_id": run_id,
        "optimizer": "base_qwen",
        "lr": "-",
        "training_time_seconds": "-",
        "maximum_memory_mb": "-",
        "avg_harness_score": format_metric(avg_harness_score),
        "training_convergence": "-",
    }


def build_task_metrics_row(run_id: str, runs_dir: Path) -> dict[str, str]:
    """Builds one task metrics row from run artifacts."""
    run_dir = runs_dir / run_id
    if run_id == BASE_QWEN_RUN_ID:
        return build_base_model_row(run_id=run_id, run_dir=run_dir)

    result_path = run_dir / "result.json"
    config_path = run_dir / "config.yaml"
    if not result_path.is_file():
        raise FileNotFoundError(f"Run result.json was not found: {result_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config.yaml was not found: {config_path}")

    result = load_json_dict(result_path)
    config = OmegaConf.load(config_path)
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError(f"Run metrics must be a dictionary: {result_path}")

    final_val_loss = find_last_history_metric(result.get("history", []), "eval_loss")
    final_train_loss = metrics.get("final_loss", None)
    convergence = final_val_loss if final_val_loss is not None else final_train_loss
    avg_harness_score = calculate_avg_harness_score(
        load_evaluation_summary_rows(run_dir / "evaluation" / "evaluation_summary.csv")
    )

    return {
        "run_id": run_id,
        "optimizer": str(config.optimizer.name),
        "lr": str(config.optimizer.lr),
        "training_time_seconds": format_metric(
            metrics.get("training_time_seconds", None),
            digits=2,
        ),
        "maximum_memory_mb": format_metric(metrics.get("max_memory_mb", None), digits=2),
        "avg_harness_score": format_metric(avg_harness_score),
        "training_convergence": format_metric(convergence),
    }


def write_task_metrics_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Writes task metrics rows to CSV."""
    columns = build_task_metrics_columns()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_task_metrics_columns() -> list[str]:
    """Returns the task metrics table column order."""
    return [
        "optimizer",
        "lr",
        "training_time_seconds",
        "maximum_memory_mb",
        "avg_harness_score",
        "training_convergence",
    ]


def escape_latex(value: str) -> str:
    """Escapes a value for LaTeX tabular output."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_task_metrics_latex(rows: list[dict[str, str]], output_path: Path) -> None:
    """Writes task metrics rows as a simple LaTeX tabular."""
    columns = build_task_metrics_columns()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        file.write("\\hline\n")
        file.write(" & ".join(escape_latex(column) for column in columns) + " \\\\\n")
        file.write("\\hline\n")
        for row in rows:
            file.write(
                " & ".join(escape_latex(row.get(column, "")) for column in columns)
                + " \\\\\n"
            )
        file.write("\\hline\n")
        file.write("\\end{tabular}\n")


def main(argv: list[str] | None = None) -> None:
    """Runs the task metrics table export CLI."""
    args = build_parser().parse_args(argv)
    runs_dir = Path(args.runs_dir)
    rows = [build_task_metrics_row(run_id=run_id, runs_dir=runs_dir) for run_id in args.run_ids]
    output_path = Path(args.output)
    write_task_metrics_csv(rows=rows, output_path=output_path)
    latex_output_path = (
        Path(args.latex_output)
        if args.latex_output is not None
        else output_path.with_suffix(".tex")
    )
    write_task_metrics_latex(rows=rows, output_path=latex_output_path)
    print(f"Task metrics table written to: {output_path}")
    print(f"Task metrics LaTeX table written to: {latex_output_path}")


if __name__ == "__main__":
    main()
