import pytest

from optimizer_comparison.optimizers.muon import Muon
from optimizer_comparison.training.muon_trainer import MuonTrainer


# /**
#  * Проверяет, что Muon optimizer пока является явной заглушкой.
#  *
#  * @return None.
#  */
def test_muon_optimizer_placeholder_rejects_use() -> None:
    with pytest.raises(NotImplementedError, match="Muon optimizer"):
        Muon()


# /**
#  * Проверяет, что MuonTrainer пока не создает optimizer.
#  *
#  * @return None.
#  */
def test_muon_trainer_placeholder_rejects_optimizer_creation() -> None:
    trainer = object.__new__(MuonTrainer)

    with pytest.raises(NotImplementedError, match="MuonTrainer optimizer path"):
        trainer.create_optimizer()
