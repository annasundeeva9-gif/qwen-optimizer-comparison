import csv
import importlib
import json
from pathlib import Path


# /**
#  * Загружает report-модуль task metrics table.
#  *
#  * @return Загруженный Python-модуль из optimizer_comparison.reports.
#  */
def load_script_module():
    return importlib.import_module("optimizer_comparison.reports.export_task_metrics_table")


# /**
#  * Создает минимальные artifacts одного training/eval run-а.
#  *
#  * @param runs_dir Директория outputs/runs для теста.
#  * @param run_id Идентификатор run-а.
#  * @return None.
#  */
def create_run_artifacts(runs_dir: Path, run_id: str) -> None:
    run_dir = runs_dir / run_id
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)

    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "optimizer:",
                "  name: adamw",
                "  lr: 0.00001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "final_loss": 2.5,
                    "training_time_seconds": 10.126,
                    "max_memory_mb": 2048.129,
                },
                "history": [
                    {"step": 1, "eval_loss": 2.4},
                    {"step": 2, "eval_loss": 2.3},
                ],
            }
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "evaluation_summary.csv").write_text(
        "\n".join(
            [
                "task,alias,metric,value,stderr",
                'piqa,piqa,"acc,none",0.7,0.01',
                'piqa,piqa,"acc_norm,none",0.8,0.01',
                'arc_easy,arc_easy,"acc,none",0.6,0.02',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


# /**
#  * Создает минимальные artifacts baseline evaluation без training result.json.
#  *
#  * @param runs_dir Директория outputs/runs для теста.
#  * @return None.
#  */
def create_base_model_artifacts(runs_dir: Path) -> None:
    evaluation_dir = runs_dir / "__base_qwen_2_5_0_5b__" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "evaluation_summary.csv").write_text(
        "\n".join(
            [
                "task,alias,metric,value,stderr",
                'piqa,piqa,"acc,none",0.75,0.01',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


# /**
#  * Проверяет сбор таблицы по критериям исходного задания.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_export_task_metrics_table_writes_expected_columns(tmp_path: Path) -> None:
    export_task_metrics_table = load_script_module()
    runs_dir = tmp_path / "runs"
    output_path = tmp_path / "reports" / "task_metrics.csv"
    latex_output_path = tmp_path / "reports" / "task_metrics.tex"
    create_run_artifacts(runs_dir=runs_dir, run_id="run-1")

    export_task_metrics_table.main(
        [
            "run-1",
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(output_path),
            "--latex-output",
            str(latex_output_path),
        ]
    )

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "optimizer": "adamw",
            "lr": "1e-05",
            "training_time_seconds": "10.13",
            "maximum_memory_mb": "2048.13",
            "avg_harness_score": "0.7000",
            "training_convergence": "2.3000",
        }
    ]
    latex_text = latex_output_path.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in latex_text
    assert "training\\_time\\_seconds" in latex_text


# /**
#  * Проверяет добавление baseline Qwen evaluation без training-метрик.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_export_task_metrics_table_supports_base_model_row(tmp_path: Path) -> None:
    export_task_metrics_table = load_script_module()
    runs_dir = tmp_path / "runs"
    output_path = tmp_path / "reports" / "task_metrics.csv"
    latex_output_path = tmp_path / "reports" / "task_metrics.tex"
    create_base_model_artifacts(runs_dir=runs_dir)

    export_task_metrics_table.main(
        [
            "__base_qwen_2_5_0_5b__",
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(output_path),
            "--latex-output",
            str(latex_output_path),
        ]
    )

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "optimizer": "base_qwen",
            "lr": "-",
            "training_time_seconds": "-",
            "maximum_memory_mb": "-",
            "avg_harness_score": "0.7500",
            "training_convergence": "-",
        }
    ]
    assert "base\\_qwen" in latex_output_path.read_text(encoding="utf-8")
