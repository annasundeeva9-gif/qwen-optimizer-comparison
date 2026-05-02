import json
from pathlib import Path

import pytest

from optimizer_comparison.evaluation.result_parser import (
    flatten_lm_eval_metrics,
    parse_lm_eval_results,
)


# /**
#  * Проверяет flatten scalar metrics из results секции harness-а.
#  *
#  * @return None.
#  */
def test_flatten_lm_eval_metrics_keeps_scalar_values() -> None:
    metrics = flatten_lm_eval_metrics(
        {
            "piqa": {
                "acc,none": 0.75,
                "acc_stderr,none": 0.01,
                "alias": "piqa",
                "flag": True,
            }
        }
    )

    assert metrics == {
        "eval_harness/piqa/acc/none": 0.75,
        "eval_harness/piqa/acc_stderr/none": 0.01,
    }


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
        "eval_harness/piqa/acc_stderr/none": 0.01,
    }
    assert parsed["versions"] == {"piqa": 1}
    assert parsed["n-samples"] == {"piqa": {"original": 100, "effective": 32}}
    assert parsed["config"] == {"batch_size": "auto", "limit": 32}


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
