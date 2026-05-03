from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from optimizer_comparison.evaluation import harness_runner
from optimizer_comparison.evaluation.harness_runner import (
    build_lm_eval_command,
    normalize_lm_eval_output_path,
    run_lm_eval_harness,
    write_raw_lm_eval_log,
)
from optimizer_comparison.evaluation.model_source import EvaluationModelSource


# /**
#  * Создает минимальный config для lm-evaluation-harness runner.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @param limit Evaluation sample limit или None.
#  * @return Hydra-like config.
#  */
def make_config(tmp_path: Path, limit: int | None) -> Any:
    return OmegaConf.create(
        {
            "mode": {"name": "smoke", "limit_eval_samples": limit},
            "evaluation": {
                "harness": {
                    "tasks": ["piqa", "arc_easy"],
                    "batch_size": 1,
                    "output_path": str(tmp_path / "lm_eval_results.json"),
                    "raw_log_path": str(tmp_path / "lm_eval_stdout.txt"),
                }
            },
        }
    )


# /**
#  * Создает model source для runner-тестов.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return EvaluationModelSource с local paths.
#  */
def make_model_source(tmp_path: Path) -> EvaluationModelSource:
    return EvaluationModelSource(
        source_type="local",
        run_dir=tmp_path,
        model_path=tmp_path / "model",
        tokenizer_path=tmp_path / "tokenizer",
    )


# /**
#  * Проверяет команду lm-evaluation-harness для smoke evaluation.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_lm_eval_command_uses_hf_model_and_cuda(tmp_path: Path) -> None:
    command = build_lm_eval_command(
        config=make_config(tmp_path=tmp_path, limit=32),
        model_source=make_model_source(tmp_path),
    )

    assert command[:5] == ["lm-eval", "run", "--model", "hf", "--model_args"]
    assert f"pretrained={tmp_path / 'model'}" in command
    assert f"tokenizer={tmp_path / 'tokenizer'}" in command
    assert "--tasks" in command
    assert "piqa" in command
    assert "arc_easy" in command
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--batch_size") + 1] == "1"
    assert command[command.index("--output_path") + 1] == str(
        tmp_path / "evaluation" / "lm_eval_results.json"
    )
    assert command[command.index("--limit") + 1] == "32"


# /**
#  * Проверяет, что full evaluation не получает --limit.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_build_lm_eval_command_omits_limit_when_not_set(tmp_path: Path) -> None:
    command = build_lm_eval_command(
        config=make_config(tmp_path=tmp_path, limit=None),
        model_source=make_model_source(tmp_path),
    )

    assert "--limit" not in command


# /**
#  * Проверяет, что runner требует CUDA до запуска subprocess.
#  *
#  * @param monkeypatch Инструмент pytest для подмены CUDA.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_lm_eval_harness_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(harness_runner.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_lm_eval_harness(
            config=make_config(tmp_path=tmp_path, limit=32),
            model_source=make_model_source(tmp_path),
        )


# /**
#  * Проверяет запись stdout и stderr в raw log.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_write_raw_lm_eval_log_writes_stdout_and_stderr(tmp_path: Path) -> None:
    raw_log_path = tmp_path / "raw.txt"

    write_raw_lm_eval_log(raw_log_path=raw_log_path, stdout="out", stderr="err")

    assert raw_log_path.read_text(encoding="utf-8") == (
        "===== STDOUT =====\nout\n===== STDERR =====\nerr"
    )


# /**
#  * Проверяет переименование timestamped JSON harness-а в стабильный файл.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_normalize_lm_eval_output_path_renames_timestamped_json(tmp_path: Path) -> None:
    stable_path = tmp_path / "lm_eval_results.json"
    timestamped_path = tmp_path / "lm_eval_results_2026-05-01T10-00-00.json"
    timestamped_path.write_text('{"results": {}}', encoding="utf-8")

    result_path = normalize_lm_eval_output_path(stable_path)

    assert result_path == stable_path
    assert stable_path.read_text(encoding="utf-8") == '{"results": {}}'
    assert not timestamped_path.exists()


# /**
#  * Проверяет, что новый timestamped JSON заменяет стабильный результат прошлого запуска.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_normalize_lm_eval_output_path_prefers_new_timestamped_json(tmp_path: Path) -> None:
    stable_path = tmp_path / "lm_eval_results.json"
    timestamped_path = tmp_path / "lm_eval_results_2026-05-01T10-00-00.json"
    stable_path.write_text('{"old": true}', encoding="utf-8")
    timestamped_path.write_text('{"new": true}', encoding="utf-8")

    result_path = normalize_lm_eval_output_path(stable_path)

    assert result_path == stable_path
    assert stable_path.read_text(encoding="utf-8") == '{"new": true}'
    assert not timestamped_path.exists()


# /**
#  * Проверяет успешный запуск runner-а через monkeypatch subprocess.run.
#  *
#  * @param monkeypatch Инструмент pytest для подмены CUDA и subprocess.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_lm_eval_harness_returns_output_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run_process(
        command: list[str],
        raw_log_path: str | Path,
        env: dict[str, str],
    ) -> int:
        calls.append(command)
        timestamped_path = tmp_path / "evaluation" / "lm_eval_results_2026.json"
        timestamped_path.parent.mkdir(parents=True, exist_ok=True)
        timestamped_path.write_text('{"results": {}}', encoding="utf-8")
        Path(raw_log_path).write_text("===== OUTPUT =====\nok", encoding="utf-8")
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUTF8"] == "1"
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["TQDM_DISABLE"] == "0"
        return 0

    monkeypatch.setattr(harness_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(harness_runner, "run_process_with_live_log", fake_run_process)

    result = run_lm_eval_harness(
        config=make_config(tmp_path=tmp_path, limit=32),
        model_source=make_model_source(tmp_path),
    )

    assert result["output_path"] == str(tmp_path / "evaluation" / "lm_eval_results.json")
    assert result["raw_log_path"] == str(tmp_path / "evaluation" / "lm_eval_stdout.txt")
    assert calls[0][0:2] == ["lm-eval", "run"]
    assert (tmp_path / "evaluation" / "lm_eval_stdout.txt").read_text(
        encoding="utf-8"
    ).startswith("===== OUTPUT =====\nok")


# /**
#  * Проверяет ошибку runner-а при ненулевом exit code.
#  *
#  * @param monkeypatch Инструмент pytest для подмены CUDA и subprocess.
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_run_lm_eval_harness_raises_on_failed_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_process(
        command: list[str],
        raw_log_path: str | Path,
        env: dict[str, str],
    ) -> int:
        Path(raw_log_path).write_text("failed", encoding="utf-8")
        return 2

    monkeypatch.setattr(harness_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(harness_runner, "run_process_with_live_log", fake_run_process)

    with pytest.raises(RuntimeError, match="exit code 2"):
        run_lm_eval_harness(
            config=make_config(tmp_path=tmp_path, limit=32),
            model_source=make_model_source(tmp_path),
        )

    assert "failed" in (tmp_path / "evaluation" / "lm_eval_stdout.txt").read_text(
        encoding="utf-8"
    )
