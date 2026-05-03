import pytest
import torch
from omegaconf import OmegaConf

from optimizer_comparison.optimizers.muon import (
    Muon,
    expand_muon_param_patterns,
)
from optimizer_comparison.training import muon_trainer as muon_trainer_module
from optimizer_comparison.training.muon_trainer import MuonTrainer


# /**
#  * Создает минимальный optimizer config для Muon.
#  *
#  * @return Hydra-like config.
#  */
def make_muon_config():
    return OmegaConf.create(
        {
            "lr": 1e-3,
            "weight_decay": 0.1,
            "muon_layer_count": 2,
            "muon_param_patterns": [
                "model.layers.{layer}.self_attn.q_proj.weight",
                "model.layers.{layer}.mlp.down_proj.weight",
            ],
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "adamw_betas": [0.9, 0.95],
            "adamw_eps": 1e-8,
        }
    )


# /**
#  * Проверяет простое раскрытие шаблонов Muon-параметров по слоям.
#  *
#  * @return None.
#  */
def test_expand_muon_param_patterns_replaces_layer_placeholder() -> None:
    assert expand_muon_param_patterns(
        layer_count=2,
        patterns=[
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.mlp.down_proj.weight",
        ],
    ) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]


# /**
#  * Проверяет, что Muon init считывает параметры из optimizer config.
#  *
#  * @return None.
#  */
def test_muon_init_reads_config_values() -> None:
    muon_param = torch.nn.Parameter(torch.ones(1, 1))
    adamw_param = torch.nn.Parameter(torch.ones(1, 1))

    optimizer = Muon(
        named_params=[
            ("model.layers.0.self_attn.q_proj.weight", muon_param),
            ("model.embed_tokens.weight", adamw_param),
        ],
        config=make_muon_config(),
    )

    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.1
    assert optimizer.muon_param_names == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }
    assert optimizer.defaults["momentum"] == 0.95
    assert optimizer.defaults["nesterov"] is True
    assert optimizer.defaults["ns_steps"] == 5
    assert optimizer.defaults["adamw_betas"] == (0.9, 0.95)
    assert optimizer.defaults["adamw_eps"] == 1e-8


# /**
#  * Проверяет, что MuonTrainer создает Muon через сохраненный optimizer config.
#  *
#  * @param monkeypatch Инструмент pytest для подмены Muon.
#  * @return None.
#  */
def test_muon_trainer_create_optimizer_builds_muon(monkeypatch: pytest.MonkeyPatch) -> None:
    created_configs = []
    fake_optimizer = object()
    trainer = object.__new__(MuonTrainer)
    trainer.optimizer_config = make_muon_config()
    trainer.optimizer = None
    model = torch.nn.Linear(1, 1)

    def fake_muon(named_params, config):
        list(named_params)
        created_configs.append(config)
        return fake_optimizer

    monkeypatch.setattr(muon_trainer_module, "Muon", fake_muon)

    optimizer = trainer.create_optimizer(model=model)

    assert optimizer is fake_optimizer
    assert trainer.optimizer is fake_optimizer
    assert created_configs == [trainer.optimizer_config]
