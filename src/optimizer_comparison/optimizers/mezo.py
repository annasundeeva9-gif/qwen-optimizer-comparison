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
            raise ValueError("optimizer.non_diff must be false. Non-differentiable MeZO is not supported.")
        self.args = args
        self.named_parameters_to_optim: list[tuple[str, torch.nn.Parameter]] = []
        self.zo_random_seed: int | None = None
        self.projected_grad: float | None = None

    def zo_perturb_parameters(self, random_seed: int | None = None, scaling_factor: float = 1) -> None:
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update.
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

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
        loss1 = self.zo_forward(model, inputs, loss_function)

        # Second function evaluation.
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs, loss_function)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # Reset model back to its parameters at start of step.
        self.zo_perturb_parameters(scaling_factor=1)

        return loss1

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
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - learning_rate * (self.projected_grad * z + self.args.weight_decay * param.data)
            else:
                param.data = param.data - learning_rate * (self.projected_grad * z)

    # def zo_forward_nondiff(self, model, inputs):
    #     """
    #     Get (no gradient) non-diffiable loss from the model.
    #     """
    #     raise RuntimeError("Don't use in AdamW/Muon/MeZO experiment")
    #     model.eval()
    #     assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

    #     with torch.inference_mode():
    #         inputs = self._prepare_inputs(inputs)
    #         args = self.args
    #         outputs = self.model.generate(
    #             inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
    #             num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
    #             num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
    #         )
    #         output_text = []
    #         for i in range(len(outputs)):
    #             output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
    #         f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

    #     return -torch.tensor(np.mean(f1s), dtype=torch.float32)
