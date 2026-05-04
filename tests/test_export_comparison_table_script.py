import csv
import importlib
import json
from pathlib import Path


# /**
#  * Загружает report-модуль comparison table.
#  *
#  * @return Загруженный Python-модуль из optimizer_comparison.reports.
#  */
def load_script_module():
    return importlib.import_module("optimizer_comparison.reports.export_comparison_table")


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
                "  name: muon",
                "  lr: 0.0001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "metrics": {"final_loss": 2.5},
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
                'arc_easy,arc_easy,"acc_norm,none",0.6,0.02',
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
#  * Проверяет сбор comparison CSV из training и evaluation artifacts.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_export_comparison_table_writes_expected_columns(tmp_path: Path) -> None:
    export_comparison_table = load_script_module()
    runs_dir = tmp_path / "runs"
    output_path = tmp_path / "reports" / "comparison.csv"
    latex_output_path = tmp_path / "reports" / "comparison.tex"
    create_run_artifacts(runs_dir=runs_dir, run_id="run-1")

    export_comparison_table.main(
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
            "optimizer": "muon",
            "lr": "0.0001",
            "train_loss": "2.5000",
            "val_loss": "2.3000",
            "arc_easy_acc_norm_none": "0.6000",
            "piqa_acc_none": "0.7000",
        }
    ]
    latex_text = latex_output_path.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in latex_text
    assert "piqa\\_acc\\_none" in latex_text


# /**
#  * Проверяет добавление baseline Qwen evaluation без train loss и val loss.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_export_comparison_table_supports_base_model_row(tmp_path: Path) -> None:
    export_comparison_table = load_script_module()
    runs_dir = tmp_path / "runs"
    output_path = tmp_path / "reports" / "comparison.csv"
    latex_output_path = tmp_path / "reports" / "comparison.tex"
    create_base_model_artifacts(runs_dir=runs_dir)

    export_comparison_table.main(
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
            "train_loss": "-",
            "val_loss": "-",
            "piqa_acc_none": "0.7500",
        }
    ]
    assert "base\\_qwen" in latex_output_path.read_text(encoding="utf-8")
