"""Export training loss and grad norm plots for a selected run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeGuard

from omegaconf import OmegaConf


# /**
#  * Создает parser для сохранения графиков train/eval метрик одного run-а.
#  *
#  * @return Parser с run_id, директорией run-ов, подписью и путями к PNG.
#  */
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export training loss and grad norm plots.")
    parser.add_argument("run_id")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--loss-output", default=None)
    parser.add_argument("--grad-norm-output", default=None)
    parser.add_argument("--label", default=None)
    return parser


# /**
#  * Проверяет, что значение можно использовать как числовую точку графика.
#  *
#  * @param value Значение из Trainer log history.
#  * @return True, если значение является числом, но не bool.
#  */
def is_plot_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


# /**
#  * Загружает JSON-файл как словарь.
#  *
#  * @param path Путь к JSON-файлу.
#  * @return Словарь из JSON-файла.
#  */
def load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


# /**
#  * Собирает пошаговые серии train/loss, eval/loss и train/grad_norm из history.
#  *
#  * @param history Список записей Trainer log_history.
#  * @return Словарь серий, где каждая серия содержит пары step/value.
#  */
def collect_training_metric_series(
    history: list[object],
) -> dict[str, list[tuple[int, float]]]:
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


# /**
#  * Возвращает true, если хотя бы одна указанная серия содержит точки для графика.
#  *
#  * @param series Словарь пошаговых серий.
#  * @param metric_names Имена серий, которые должны быть проверены.
#  * @return True, если график имеет смысл сохранять.
#  */
def has_plot_points(
    series: dict[str, list[tuple[int, float]]],
    metric_names: list[str],
) -> bool:
    return any(series.get(metric_name, []) for metric_name in metric_names)


# /**
#  * Определяет подпись графика из CLI или optimizer config.
#  *
#  * @param config Hydra config run-а.
#  * @param custom_label Явная подпись из CLI.
#  * @param run_id Идентификатор run-а из CLI.
#  * @return Подпись для заголовка графика.
#  */
def resolve_plot_label(config: Any, custom_label: str | None, run_id: str) -> str:
    if custom_label:
        return custom_label

    optimizer_name = OmegaConf.select(config, "optimizer.name")
    if isinstance(optimizer_name, str) and optimizer_name:
        return optimizer_name

    return run_id


# /**
#  * Сохраняет один PNG-график по выбранным пошаговым сериям.
#  *
#  * @param series Словарь всех доступных пошаговых серий.
#  * @param metric_names Имена серий, которые нужно нарисовать.
#  * @param output_path Путь к PNG-файлу.
#  * @param title Заголовок графика.
#  * @param ylabel Подпись оси Y.
#  * @param styles Стили matplotlib для отдельных серий.
#  * @param empty_error Сообщение об ошибке, если точек нет.
#  * @return Путь к сохраненному PNG.
#  */
def save_metric_plot(
    series: dict[str, list[tuple[int, float]]],
    metric_names: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
    styles: dict[str, dict[str, Any]],
    empty_error: str,
) -> Path:
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


# /**
#  * Сохраняет два PNG-графика: loss-кривые и train/grad_norm.
#  *
#  * @param series Словарь серий train/loss, eval/loss и train/grad_norm.
#  * @param loss_output_path Путь к PNG-файлу с loss-кривыми.
#  * @param grad_norm_output_path Путь к PNG-файлу с grad_norm.
#  * @param label Подпись модели/оптимизатора для заголовка.
#  * @param lr Learning rate из optimizer config.
#  * @return Пара путей к сохраненным PNG.
#  */
def save_training_metrics_plot(
    series: dict[str, list[tuple[int, float]]],
    loss_output_path: Path,
    grad_norm_output_path: Path,
    label: str,
    lr: object,
) -> tuple[Path, Path]:
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


# /**
#  * Собирает и сохраняет графики пошаговых train/eval метрик для одного run-а.
#  *
#  * @param run_id Идентификатор run-а внутри outputs/runs.
#  * @param runs_dir Директория с run artifacts.
#  * @param output_dir Директория для стандартных PNG-путей.
#  * @param loss_output_path Явный путь к loss PNG или None для стандартного пути.
#  * @param grad_norm_output_path Явный путь к grad_norm PNG или None для стандартного пути.
#  * @param custom_label Явная подпись графиков или None для optimizer.name.
#  * @return Пара путей к сохраненным PNG.
#  */
def export_training_metrics_plot(
    run_id: str,
    runs_dir: Path,
    output_dir: Path,
    loss_output_path: Path | None,
    grad_norm_output_path: Path | None,
    custom_label: str | None,
) -> tuple[Path, Path]:
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


# /**
#  * Точка входа CLI для построения PNG-графиков train/eval метрик.
#  *
#  * @param argv CLI-аргументы или None для чтения из sys.argv.
#  * @return None.
#  */
def main(argv: list[str] | None = None) -> None:
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
