from pathlib import Path

from optimizer_comparison.training.plots import (
    add_training_curves_artifact,
    collect_training_curve_series,
    save_training_curves_plot,
)
from optimizer_comparison.training.result_contract import build_training_result


# /**
#  * Проверяет сбор точек train loss, validation loss и learning rate из history.
#  *
#  * @return None.
#  */
def test_collect_training_curve_series_reads_expected_metrics() -> None:
    series = collect_training_curve_series(
        [
            {"step": 1, "loss": 2.0, "learning_rate": 1e-5},
            {"step": 2, "eval_loss": 1.5},
            {"step": 3, "loss": 1.8, "eval_loss": 1.4, "learning_rate": 8e-6},
            {"loss": 9.0},
        ]
    )

    assert series["train_loss"] == [(1, 2.0), (3, 1.8)]
    assert series["val_loss"] == [(2, 1.5), (3, 1.4)]
    assert series["learning_rate"] == [(1, 1e-5), (3, 8e-6)]


# /**
#  * Проверяет сохранение PNG-графика training curves.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_save_training_curves_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "training_curves.png"

    saved_path = save_training_curves_plot(
        history=[
            {"step": 1, "loss": 2.0, "learning_rate": 1e-5},
            {"step": 2, "eval_loss": 1.5},
        ],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG")


# /**
#  * Проверяет, что пустая history не создает пустой график.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_save_training_curves_plot_skips_empty_history(tmp_path: Path) -> None:
    output_path = tmp_path / "training_curves.png"

    saved_path = save_training_curves_plot(history=[], output_path=output_path)

    assert saved_path is None
    assert not output_path.exists()


# /**
#  * Проверяет добавление пути training-графика в training-result artifacts.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_add_training_curves_artifact_updates_result(tmp_path: Path) -> None:
    result = build_training_result(
        run_name="adamw_baseline",
        status="completed",
        final_loss=1.0,
        training_time_seconds=10.0,
        time_per_step_seconds=1.0,
        max_memory_mb=100.0,
    )
    result["history"] = [{"step": 1, "loss": 2.0, "learning_rate": 1e-5}]

    updated_result = add_training_curves_artifact(result=result, run_dir=tmp_path)

    artifacts = updated_result["artifacts"]
    assert isinstance(artifacts, dict)
    plots = artifacts["plots"]
    assert isinstance(plots, dict)
    assert plots["training_curves_path"] == str(tmp_path / "training_curves.png")
    assert (tmp_path / "training_curves.png").is_file()
