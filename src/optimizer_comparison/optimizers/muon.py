"""Muon optimizer placeholder."""
# This code is adapted from:
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
#
# Original implementation:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import cast, overload

import torch
from omegaconf import DictConfig


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization."""
    assert len(matrix.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16()
    if matrix.size(0) > matrix.size(1):
        x = x.T

    # Ensure spectral norm is at most 1 before Newton-Schulz iterations.
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram_matrix = x @ x.T
        update_matrix = b * gram_matrix + c * gram_matrix @ gram_matrix
        x = a * x + update_matrix @ x

    if matrix.size(0) > matrix.size(1):
        x = x.T
    return cast(torch.Tensor, x)

def expand_muon_param_patterns(layer_count: int, patterns: list[str]) -> list[str]:
    """Expands Muon parameter name patterns for all model layers."""
    muon_params: list[str] = []
    for layer_index in range(layer_count):
        for pattern in patterns:
            muon_params.append(pattern.format(layer=layer_index))
    return muon_params

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.nn.Parameter]],
        config: DictConfig,
    ) -> None:
        adamw_betas = list(config.adamw_betas)
        if len(adamw_betas) != 2:
            raise ValueError("optimizer.adamw_betas must contain exactly two values.")

        self.muon_param_names = set(expand_muon_param_patterns(
            layer_count=int(config.muon_layer_count),
            patterns=[str(pattern) for pattern in config.muon_param_patterns],
        ))

        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []

        for name, param in named_params:
            if not param.requires_grad:
                continue

            if name in self.muon_param_names:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self.weight_decay = float(config.weight_decay)

        defaults = {
            "lr": float(config.lr),
            "weight_decay": self.weight_decay,
            "momentum": float(config.momentum),
            "nesterov": bool(config.nesterov),
            "ns_steps": int(config.ns_steps),
            "adamw_betas": (float(adamw_betas[0]), float(adamw_betas[1])),
            "adamw_eps": float(config.adamw_eps),
        }

        param_groups = [
            {**defaults, "params": muon_params, "use_muon": True},
            {**defaults, "params": adamw_params, "use_muon": False},
        ]

        super().__init__(param_groups, defaults)

    def adjust_lr_for_muon(self, lr: float, param_shape: torch.Size) -> float:
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr
    
    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf

                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                    p.mul_(1 - lr * weight_decay)
                    p.add_(u, alpha=-adjusted_lr)

            else:
                lr = group["lr"]
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)

                    state["step"] += 1
                    step = state["step"]

                    buf1 = state["moment1"]
                    buf2 = state["moment2"]

                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5

                    p.mul_(1 - lr * weight_decay)
                    p.add_(g, alpha=-lr / scale)

        return loss
