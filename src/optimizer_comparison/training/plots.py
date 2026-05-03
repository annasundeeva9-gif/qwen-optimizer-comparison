"""Training plot artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard


# /**
#  * Проверяет, что значение можно использовать как числовую точку на графике.
#  *
#  * @param value Значение из Trainer log history.
#  * @return True, если значение является числом, но не bool.
#  */
def is_plot_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


# /**
#  * Собирает точки train loss, validation loss и learning rate из Trainer log history.
#  *
#  * @param history Список записей Trainer log_history.
#  * @return Словарь серий, где каждая серия содержит пары step/value.
#  */
def collect_training_curve_series(
    history: list[object],
) -> dict[str, list[tuple[int, float]]]:
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


# /**
#  * Возвращает true, если хотя бы одна серия содержит точки для графика.
#  *
#  * @param series Словарь серий training-графика.
#  * @return True, если график имеет смысл сохранять.
#  */
def has_training_curve_points(series: dict[str, list[tuple[int, float]]]) -> bool:
    return any(points for points in series.values())


# /**
#  * Сохраняет один PNG-график train loss, validation loss и learning rate.
#  *
#  * @param history Список записей Trainer log_history.
#  * @param output_path Путь к PNG-файлу.
#  * @return Путь к файлу или None, если в history нет подходящих точек.
#  */
def save_training_curves_plot(
    history: list[object],
    output_path: str | Path,
) -> Path | None:
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


# /**
#  * Добавляет путь training-графика в result artifacts, если график удалось построить.
#  *
#  * @param result Training-result с history и artifacts.
#  * @param run_dir Директория run-а для сохранения PNG.
#  * @return Тот же training-result с обновленной секцией artifacts.plots.
#  */
def add_training_curves_artifact(
    result: dict[str, object],
    run_dir: str | Path,
) -> dict[str, object]:
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
