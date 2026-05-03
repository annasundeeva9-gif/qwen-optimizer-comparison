"""Muon trainer with custom optimizer creation."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import Trainer

from optimizer_comparison.optimizers.muon import Muon


# /**
#  * Trainer с переопределенным созданием Muon optimizer.
#  */
class MuonTrainer(Trainer):
    # /**
    #  * Сохраняет optimizer config для последующего создания Muon.
    #  *
    #  * @param optimizer_config Hydra optimizer config.
    #  * @param kwargs Аргументы базового HuggingFace Trainer.
    #  */
    def __init__(self, optimizer_config: DictConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.optimizer_config = optimizer_config

    # /**
    #  * Создает Muon optimizer из Hydra config.
    #  *
    #  * @param model Модель, для которой Trainer создает optimizer.
    #  * @return Optimizer, созданный через Muon.
    #  */
    def create_optimizer(self, model: Any = None) -> Optimizer:
        if self.optimizer is None:
            if model is None:
                model = self.model

            self.optimizer = Muon(
                named_params=model.named_parameters(),
                config=self.optimizer_config,
            )

        return self.optimizer
