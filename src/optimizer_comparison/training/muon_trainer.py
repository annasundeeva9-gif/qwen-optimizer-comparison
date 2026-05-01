"""Muon trainer placeholder."""

from transformers import Trainer


# /**
#  * Заглушка для будущего Trainer с переопределенным Muon optimizer path.
#  */
class MuonTrainer(Trainer):
    # /**
    #  * Запрещает случайное создание Muon optimizer до ручной интеграции.
    #  *
    #  * @return None.
    #  */
    def create_optimizer(self) -> None:
        raise NotImplementedError("MuonTrainer optimizer path is pending manual integration.")
