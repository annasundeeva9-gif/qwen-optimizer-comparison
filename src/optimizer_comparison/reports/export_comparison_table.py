"""Build comparison CSV for selected training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

BASE_QWEN_RUN_ID = "__base_qwen_2_5_0_5b__"


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for comparison table export."""
    parser = argparse.ArgumentParser(description="Export comparison table for selected runs.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output", default="outputs/reports/comparison.csv")
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


def build_harness_column_name(task: str, metric: str) -> str:
    """Normalizes a harness metric name for a table column."""
    normalized_metric = (
        metric.replace(",", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )
    return f"{task}_{normalized_metric}".lower()


def load_harness_metrics(summary_path: Path) -> dict[str, str]:
    """Loads harness metrics from one evaluation summary CSV."""
    if not summary_path.is_file():
        return {}

    metrics: dict[str, str] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            task = row.get("task", "")
            metric = row.get("metric", "")
            value = row.get("value", "")
            if task and metric:
                metrics[build_harness_column_name(task=task, metric=metric)] = format_table_number(
                    value
                )
    return metrics


def format_table_number(value: object) -> str:
    """Formats a numeric table cell with four decimal places."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    if isinstance(value, str):
        try:
            return f"{float(value):.4f}"
        except ValueError:
            return value
    return ""


def build_base_model_row(run_id: str, run_dir: Path) -> dict[str, str]:
    """Builds the comparison row for the base Qwen evaluation."""
    row = {
        "run_id": run_id,
        "optimizer": "base_qwen",
        "lr": "-",
        "train_loss": "-",
        "val_loss": "-",
    }
    row.update(load_harness_metrics(run_dir / "evaluation" / "evaluation_summary.csv"))
    return row


def build_comparison_row(run_id: str, runs_dir: Path) -> dict[str, str]:
    """Builds one comparison row from run artifacts."""
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

    final_loss = metrics.get("final_loss", "")
    train_loss = format_table_number(final_loss)
    val_loss = find_last_history_metric(result.get("history", []), "eval_loss")

    row = {
        "run_id": run_id,
        "optimizer": str(config.optimizer.name),
        "lr": str(config.optimizer.lr),
        "train_loss": train_loss,
        "val_loss": "" if val_loss is None else format_table_number(val_loss),
    }
    row.update(load_harness_metrics(run_dir / "evaluation" / "evaluation_summary.csv"))
    return row


def write_comparison_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Writes comparison rows to CSV."""
    columns = build_comparison_columns(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_comparison_columns(rows: list[dict[str, str]]) -> list[str]:
    """Builds the comparison table column order."""
    base_columns = ["optimizer", "lr", "train_loss", "val_loss"]
    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_columns and key != "run_id"
        }
    )
    return base_columns + extra_columns


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


def write_comparison_latex(rows: list[dict[str, str]], output_path: Path) -> None:
    """Writes comparison rows as a simple LaTeX tabular."""
    columns = build_comparison_columns(rows)
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
    """Runs the comparison table export CLI."""
    args = build_parser().parse_args(argv)
    runs_dir = Path(args.runs_dir)
    rows = [build_comparison_row(run_id=run_id, runs_dir=runs_dir) for run_id in args.run_ids]
    output_path = Path(args.output)
    write_comparison_csv(rows=rows, output_path=output_path)
    latex_output_path = (
        Path(args.latex_output)
        if args.latex_output is not None
        else output_path.with_suffix(".tex")
    )
    write_comparison_latex(rows=rows, output_path=latex_output_path)
    print(f"Comparison table written to: {output_path}")
    print(f"Comparison LaTeX table written to: {latex_output_path}")


if __name__ == "__main__":
    main()
