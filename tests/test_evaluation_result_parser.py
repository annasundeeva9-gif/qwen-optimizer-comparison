import json
from pathlib import Path

import pytest

from optimizer_comparison.evaluation.result_parser import (
    build_lm_eval_summary_rows,
    flatten_lm_eval_metrics,
    parse_lm_eval_results,
    write_lm_eval_summary_csv,
)


# /**
#  * Проверяет flatten scalar metrics из results секции harness-а.
#  *
#  * @return None.
#  */
def test_flatten_lm_eval_metrics_keeps_primary_scalar_values() -> None:
    metrics = flatten_lm_eval_metrics(
        {
            "piqa": {
                "acc,none": 0.75,
                "acc_stderr,none": 0.01,
                "acc_norm,none": 0.76,
                "acc_norm_stderr,none": 0.02,
                "alias": "piqa",
                "flag": True,
            }
        }
    )

    assert metrics == {
        "eval_harness/piqa/acc/none": 0.75,
        "eval_harness/piqa/acc_norm/none": 0.76,
    }


# /**
#  * Проверяет строки CSV summary с основными metric и stderr-колонкой.
#  *
#  * @return None.
#  */
def test_build_lm_eval_summary_rows_pairs_metric_with_stderr() -> None:
    rows = build_lm_eval_summary_rows(
        {
            "piqa": {
                "acc,none": 0.75,
                "acc_stderr,none": 0.01,
                "alias": "PIQA",
            }
        }
    )

    assert rows == [
        {
            "task": "piqa",
            "alias": "PIQA",
            "metric": "acc,none",
            "value": "0.75",
            "stderr": "0.01",
        }
    ]


# /**
#  * Проверяет парсинг оригинального JSON lm-evaluation-harness.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_parse_lm_eval_results_returns_metrics_and_metadata(tmp_path: Path) -> None:
    result_path = tmp_path / "lm_eval_results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {
                    "piqa": {
                        "acc,none": 0.75,
                        "acc_stderr,none": 0.01,
                        "alias": "piqa",
                    }
                },
                "versions": {"piqa": 1},
                "n-shot": {"piqa": 0},
                "n-samples": {"piqa": {"original": 100, "effective": 32}},
                "higher_is_better": {"piqa": {"acc": True}},
                "config": {"batch_size": "auto", "limit": 32},
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_lm_eval_results(result_path)

    assert parsed["result_path"] == str(result_path)
    assert parsed["tasks"] == ["piqa"]
    assert parsed["metrics"] == {
        "eval_harness/piqa/acc/none": 0.75,
    }
    assert parsed["summary_rows"][0]["stderr"] == "0.01"
    assert parsed["versions"] == {"piqa": 1}
    assert parsed["n-samples"] == {"piqa": {"original": 100, "effective": 32}}
    assert parsed["config"] == {"batch_size": "auto", "limit": 32}


# /**
#  * Проверяет запись CSV summary рядом с результатом lm-evaluation-harness.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_write_lm_eval_summary_csv_writes_table(tmp_path: Path) -> None:
    result_path = tmp_path / "lm_eval_results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {
                    "piqa": {
                        "acc,none": 0.75,
                        "acc_stderr,none": 0.01,
                        "alias": "PIQA",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    csv_path = write_lm_eval_summary_csv(result_path)

    assert csv_path == tmp_path / "evaluation_summary.csv"
    assert csv_path.read_text(encoding="utf-8") == (
        "task,alias,metric,value,stderr\n"
        "piqa,PIQA,\"acc,none\",0.75,0.01\n"
    )


# /**
#  * Проверяет понятную ошибку для JSON без results.
#  *
#  * @param tmp_path Временная директория pytest.
#  * @return None.
#  */
def test_parse_lm_eval_results_requires_results_section(tmp_path: Path) -> None:
    result_path = tmp_path / "lm_eval_results.json"
    result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="results"):
        parse_lm_eval_results(result_path)
