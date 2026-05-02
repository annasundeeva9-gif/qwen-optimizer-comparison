"""Muon trainer placeholder."""

from typing import Any

from torch.optim import Optimizer
from transformers import Trainer


# /**
#  * Заглушка для будущего Trainer с переопределенным Muon optimizer path.
#  */
class MuonTrainer(Trainer):
    # /**
    #  * Запрещает случайное создание Muon optimizer до ручной интеграции.
    #  *
    #  * @param model Модель, для которой Trainer создает optimizer.
    #  * @return Optimizer, когда Muon будет интегрирован.
    #  */
    def create_optimizer(self, model: Any = None) -> Optimizer:
        raise NotImplementedError("MuonTrainer optimizer path is pending manual integration.")
