"""lm-evaluation-harness runner skeleton."""

from __future__ import annotations

from omegaconf import DictConfig


# /**
#  * Запускает lm-evaluation-harness и сохраняет stdout/stderr в файл.
#  *
#  * @param config Конфигурация evaluation.
#  * @return None. Результаты должны быть записаны по путям из конфига.
#  */
def run_lm_eval_harness(config: DictConfig) -> None:
    raise NotImplementedError("lm-evaluation-harness runner implementation is pending.")
