from optimizer_comparison.training import seed as seed_module
from optimizer_comparison.training.seed import set_seed


# /**
#  * Проверяет, что set_seed выставляет seed во всех используемых backend-ах.
#  *
#  * @param monkeypatch Инструмент pytest для подмены random/NumPy/Torch.
#  * @return None.
#  */
def test_set_seed_updates_python_numpy_torch_and_cuda(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(seed_module.random, "seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(seed_module.np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(
        seed_module.torch,
        "manual_seed",
        lambda seed: calls.append(("torch", seed)),
    )
    monkeypatch.setattr(seed_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        seed_module.torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(("cuda", seed)),
    )

    set_seed(42)

    assert calls == [
        ("python", 42),
        ("numpy", 42),
        ("torch", 42),
        ("cuda", 42),
    ]
