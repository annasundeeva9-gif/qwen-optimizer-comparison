"""MeZO pseudo-optimizer implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class MeZO:
    """MeZO update helper.

    This is not a full torch optimizer: MeZO needs the current batch and the
    Trainer loss function to estimate a directional gradient.
    """

    def __init__(self, args: Any) -> None:
        if bool(args.non_diff):
            raise ValueError(
                "optimizer.non_diff must be false. Non-differentiable MeZO is not supported."
            )
        self.args = args
        self.zo_eps = float(args.zo_eps)
        self.weight_decay = float(args.weight_decay)
        self.named_parameters_to_optim: list[tuple[str, torch.nn.Parameter]] = []
        self.zo_random_seed: int | None = None
        self.projected_grad: float | None = None
        self.last_debug_stats: dict[str, float] | None = None

    def zo_perturb_parameters(
        self, random_seed: int | None = None, scaling_factor: float = 1
    ) -> None:
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation.
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update.
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for _name, param in self.named_parameters_to_optim:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data + scaling_factor * z * self.zo_eps

    def zo_forward(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        loss_function: Callable[[torch.nn.Module, dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor:
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation).
            raise RuntimeError("Non implemented now")
        return loss_function(model, inputs)

    def zo_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        loss_function: Callable[[torch.nn.Module, dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor:
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z).
        """

        # What parameters to optimize.
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z.
        self.zo_random_seed = int(np.random.randint(1000000000))

        # First function evaluation.
        self.zo_perturb_parameters(scaling_factor=1)
        loss_plus = self.zo_forward(model, inputs, loss_function)

        # Second function evaluation.
        self.zo_perturb_parameters(scaling_factor=-2)
        loss_minus = self.zo_forward(model, inputs, loss_function)

        loss_diff = loss_plus - loss_minus
        projected_grad = loss_diff / (2 * self.zo_eps)
        self.projected_grad = projected_grad.detach().float().item()
        self.last_debug_stats = {
            "loss_plus": loss_plus.detach().float().item(),
            "loss_minus": loss_minus.detach().float().item(),
            "loss_diff": loss_diff.detach().float().item(),
            "projected_grad": self.projected_grad,
        }

        # Reset model back to its parameters at start of step.
        self.zo_perturb_parameters(scaling_factor=1)

        return loss_plus

    def zo_update(self, learning_rate: float) -> None:
        """
        Update the parameters with the estimated gradients.
        """
        if self.projected_grad is None:
            raise RuntimeError("zo_update called before zo_step.")

        # Reset the random seed for sampling zs.
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            # Resample z.
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - learning_rate * (
                    self.projected_grad * z + self.weight_decay * param.data
                )
            else:
                param.data = param.data - learning_rate * (self.projected_grad * z)
