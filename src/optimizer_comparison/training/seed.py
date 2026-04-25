"""Reproducibility utilities."""

from __future__ import annotations

import random


# /**
#  * Устанавливает базовый seed для воспроизводимости.
#  *
#  * @param seed Целочисленное значение seed.
#  * @return None.
#  */
def set_seed(seed: int) -> None:
    random.seed(seed)
