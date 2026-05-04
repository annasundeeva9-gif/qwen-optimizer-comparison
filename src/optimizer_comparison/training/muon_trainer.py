"""Muon trainer with custom optimizer creation."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import Trainer

from optimizer_comparison.optimizers.muon import Muon


class MuonTrainer(Trainer):
    """Trainer that creates the Muon optimizer from Hydra config."""

    def __init__(self, optimizer_config: DictConfig, **kwargs: Any) -> None:
        """Stores optimizer config for Muon creation."""
        super().__init__(**kwargs)
        self.optimizer_config = optimizer_config

    def create_optimizer(self, model: Any = None) -> Optimizer:
        """Creates the Muon optimizer for the current model."""
        if self.optimizer is None:
            if model is None:
                model = self.model

            self.optimizer = Muon(
                named_params=model.named_parameters(),
                config=self.optimizer_config,
            )

        return self.optimizer
