"""Copy selected run logs for report artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for run log export."""
    parser = argparse.ArgumentParser(description="Export selected run logs for reports.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="report/artifacts/logs")
    return parser


def copy_required_file(source_path: Path, target_path: Path) -> None:
    """Copies one required artifact file."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Required run artifact was not found: {source_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def export_run_logs(run_id: str, runs_dir: Path, output_dir: Path) -> Path:
    """Copies config, result, and evaluation summary for one run."""
    run_dir = runs_dir / run_id
    target_dir = output_dir / run_id

    copy_required_file(
        source_path=run_dir / "config.yaml",
        target_path=target_dir / "config.yaml",
    )
    copy_required_file(
        source_path=run_dir / "result.json",
        target_path=target_dir / "result.json",
    )
    copy_required_file(
        source_path=run_dir / "evaluation" / "evaluation_summary.csv",
        target_path=target_dir / "evaluation_summary.csv",
    )
    return target_dir


def main(argv: list[str] | None = None) -> None:
    """Runs the run log export CLI."""
    args = build_parser().parse_args(argv)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    for run_id in args.run_ids:
        target_dir = export_run_logs(
            run_id=run_id,
            runs_dir=runs_dir,
            output_dir=output_dir,
        )
        print(f"Run logs written to: {target_dir}")


if __name__ == "__main__":
    main()
