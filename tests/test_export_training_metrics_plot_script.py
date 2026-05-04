import importlib
import json
from pathlib import Path


# /**
#  * Загружает report-модуль training metrics plot.
#  *
#  * @return Загруженный Python-модуль из optimizer_comparison.reports.
#  */
def load_script_module():
    return importlib.import_module("optimizer_comparison.reports.export_training_metrics_plot")


# /**
#  * Создает минимальные artifacts training run-а для проверки графика.
#  *
#  * @param runs_dir Директория outputs/runs для теста.
#  * @param run_id Идентификатор run-а.
#  * @return None.
#  */
def create_run_artifacts(runs_dir: Path, run_id: str) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "optimizer:",
                "  name: adamw",
                "  lr: 0.00001",
                "experiment:",
                "  name: adamw_test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "run_name": "adamw_plot_test",
                "history": [
                    {"step": 1, "loss": 2.0, "grad_norm": 0.5},
                    {"step": 2, "eval_loss": 1.8},
                    {"step": 3, "loss": 1.9, "grad_norm": 0.4},
                ],
            }
        ),
        encoding="utf-8",
    )


# /**
#  * Проверяет сбор пошаговых серий train/loss, eval/loss и train/grad_norm.
#  *
#  * @return None.
#  */
def test_collect_training_metric_series_reads_expected_metrics() -> None:
    export_training_metrics_plot = load_script_module()

    series = export_training_metrics_plot.collect_training_metric_series(
        [
            {"step": 1, "loss": 2.0, "grad_norm": 0.5},
            {"step": 2, "eval_loss": 1.8},
            {"step": 3, "loss": 1.9, "grad_norm": 0.4},
            {"loss": 9.0, "grad_norm": 9.0},
        ]
    )

    assert series["train/loss"] == [(1, 2.0), (3, 1.9)]
    assert series["eval/loss"] == [(2, 1.8)]
    assert series["train/grad_norm"] == [(1, 0.5), (3, 0.4)]


# /**
#  * Проверяет, что CLI-логика сохраняет PNG-графики loss и grad_norm для training run-а.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_export_training_metrics_plot_writes_png(tmp_path: Path) -> None:
    export_training_metrics_plot = load_script_module()
    runs_dir = tmp_path / "runs"
    loss_output_path = tmp_path / "reports" / "losses.png"
    grad_norm_output_path = tmp_path / "reports" / "grad_norm.png"
    create_run_artifacts(runs_dir=runs_dir, run_id="run-1")

    export_training_metrics_plot.main(
        [
            "run-1",
            "--runs-dir",
            str(runs_dir),
            "--loss-output",
            str(loss_output_path),
            "--grad-norm-output",
            str(grad_norm_output_path),
            "--label",
            "AdamW custom label",
        ]
    )

    assert loss_output_path.read_bytes().startswith(b"\x89PNG")
    assert grad_norm_output_path.read_bytes().startswith(b"\x89PNG")
