"""MeZO trainer with custom optimizer creation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from accelerate import skip_first_batches
from omegaconf import DictConfig
from transformers import Trainer
from transformers.utils import is_torch_xla_available, logging

from optimizer_comparison.optimizers.mezo import MeZO

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)


class MeZOTrainer(Trainer):
    """Trainer that creates the MeZO optimizer from Hydra config."""

    def __init__(self, optimizer_config: DictConfig, **kwargs: Any) -> None:
        """Stores optimizer config for MeZO creation."""
        super().__init__(**kwargs)
        self.optimizer_config = optimizer_config
        self.mezo_pseudo_optimizer = MeZO(args=optimizer_config)
        self.mezo_debug_enabled = bool(optimizer_config.get("debug_enabled", False))
        self.mezo_debug_every = int(optimizer_config.get("debug_every", 50))
        self.mezo_debug_param_patterns = [
            str(pattern) for pattern in optimizer_config.get("debug_param_patterns", [])
        ]

    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        """Run one full pass over the dataloader."""

        assert self.args.gradient_accumulation_steps == 1

        step = -1
        grad_norm = None
        learning_rate = None
        rng_to_sync = False

        # Handle resumption from checkpoint: skip already-trained batches in the resumed epoch
        num_update_steps_trained = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
                train_dataloader = skip_first_batches(
                    train_dataloader, steps_trained_in_current_epoch
                )
                step = steps_trained_in_current_epoch - 1
                num_update_steps_trained = (
                    steps_trained_in_current_epoch // self.args.gradient_accumulation_steps
                )
                rng_to_sync = True
            elif steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        epoch_iterator = iter(train_dataloader)

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % self.args.gradient_accumulation_steps
        if remainder == 0:
            remainder = self.args.gradient_accumulation_steps

        # Outer loop: one iteration per optimizer step. Each iteration prefetches
        # `gradient_accumulation_steps` batches (fewer for the last step if the epoch
        # doesn't divide evenly).
        for update_step in range(num_update_steps_trained, num_update_steps_per_epoch):
            num_batches = (
                self.args.gradient_accumulation_steps
                if update_step != (num_update_steps_per_epoch - 1)
                else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(
                epoch_iterator, num_batches, self.args.device
            )

            # This scales the loss when the last accumulation step has fewer batches.
            # Not used if `num_items_in_batch` is not None.
            self.current_gradient_accumulation_steps = len(batch_samples)

            # Sync after skipped batches to keep the shuffled dataloader order reproducible.
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Inner loop: forward + backward for each micro-batch. Gradients are
            # accumulated without syncing until the last micro-batch, then we clip,
            # step the optimizer, and log/save/evaluate.
            for _i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        self.args, self.state, self.control
                    )

                debug_this_step = self._should_debug_mezo()
                watched_params_before = None
                debug_loss_before_update = None
                if debug_this_step:
                    watched_params_before = self._capture_mezo_watched_params(model)
                    debug_loss_before_update = (
                        self._mezo_forward_loss(model, inputs).detach().float().item()
                    )

                # MeZO added: estimate gradient
                tr_loss_step = self.mezo_pseudo_optimizer.zo_step(
                    model, inputs, self._mezo_forward_loss
                )

                restore_logs: dict[str, float] = {}
                if watched_params_before is not None:
                    restore_logs = self._build_mezo_param_diff_logs(
                        model=model,
                        watched_params_before=watched_params_before,
                        metric_prefix="debug/restore_max_diff",
                    )
                    self._log_mezo_debug_metrics(
                        {
                            **restore_logs,
                            **self._get_mezo_step_debug_logs(),
                            "debug/zo_eps": self.mezo_pseudo_optimizer.zo_eps,
                        }
                    )

                if (
                    self.args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    self._tr_loss += self._tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    if self._tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            "Calculated loss must be on the original device: "
                            f"{self._tr_loss.device}, but device in use is "
                            f"{tr_loss_step.device}"
                        )
                    self._tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self._track_num_input_tokens(inputs)

                if do_sync_step:
                    # MeZO added: update model with the estimated gradient
                    learning_rate = self._get_learning_rate()
                    if debug_this_step:
                        self._log_mezo_debug_metrics(
                            {
                                "debug/learning_rate_before_update": learning_rate,
                                **self._get_mezo_update_debug_logs(),
                            }
                        )
                    self.mezo_pseudo_optimizer.zo_update(learning_rate=learning_rate)

                    update_logs: dict[str, float] = {}
                    debug_loss_after_update = None
                    if watched_params_before is not None:
                        update_logs = self._build_mezo_param_diff_logs(
                            model=model,
                            watched_params_before=watched_params_before,
                            metric_prefix="debug/update_max_diff",
                        )
                        if debug_loss_before_update is not None:
                            debug_loss_after_update = (
                                self._mezo_forward_loss(model, inputs).detach().float().item()
                            )
                            update_logs.update(
                                {
                                    "debug/loss_before_update": debug_loss_before_update,
                                    "debug/loss_after_update": debug_loss_after_update,
                                    "debug/loss_after_minus_before": (
                                        debug_loss_after_update - debug_loss_before_update
                                    ),
                                }
                            )
                        self._log_mezo_debug_metrics(update_logs)
                        self._write_mezo_debug_rows(
                            epoch=epoch,
                            learning_rate=learning_rate,
                            restore_logs=restore_logs,
                            update_logs=update_logs,
                        )

                    self.lr_scheduler.step()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        self.args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        self._tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        self.args, self.state, self.control
                    )
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        # PyTorch/XLA relies on the dataloader to insert mark_step each iteration.
        # When we break out of the loop early, we flush the pending graph manually.
        if is_torch_xla_available():
            xm.mark_step()

        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, "
                f"stopping training at step {self.state.global_step}! This is "
                "expected if you're using an IterableDataset and set num_steps "
                f"({self.state.max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(
            self._tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

    def _mezo_forward_loss(self, model, inputs):
        """Computes a no-grad Trainer loss for one MeZO function evaluation."""
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def _should_debug_mezo(self) -> bool:
        """Checks whether MeZO debug logging should run on the current optimizer step."""
        if not self.mezo_debug_enabled:
            return False
        if self.mezo_debug_every <= 0:
            return False
        return self.state.global_step % self.mezo_debug_every == 0

    def _capture_mezo_watched_params(self, model) -> dict[str, torch.Tensor]:
        """Stores CPU snapshots of selected parameters before MeZO perturbation."""
        watched_params: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in self.mezo_debug_param_patterns):
                watched_params[name] = param.detach().float().cpu().clone()
        return watched_params

    def _build_mezo_param_diff_logs(
        self,
        model,
        watched_params_before: dict[str, torch.Tensor],
        metric_prefix: str,
    ) -> dict[str, float]:
        """Computes max parameter diffs against the pre-step snapshots."""
        logs: dict[str, float] = {}
        for name, param in model.named_parameters():
            if name not in watched_params_before:
                continue
            diff = (param.detach().float().cpu() - watched_params_before[name]).abs().max().item()
            logs[f"{metric_prefix}/{name}"] = float(diff)
        return logs

    def _get_mezo_step_debug_logs(self) -> dict[str, float | str]:
        """Builds debug logs from the latest MeZO finite-difference step."""
        stats = self.mezo_pseudo_optimizer.last_debug_stats
        logs: dict[str, float | str] = {
            "debug/param_dtype": str(next(self.model.parameters()).dtype),
        }
        if stats is None:
            return logs

        logs.update(
            {
                "debug/loss_plus": stats["loss_plus"],
                "debug/loss_minus": stats["loss_minus"],
                "debug/loss_diff": stats["loss_diff"],
                "debug/projected_grad": stats["projected_grad"],
            }
        )
        return logs

    def _get_mezo_update_debug_logs(self) -> dict[str, float]:
        """Builds debug logs used immediately before the MeZO update."""
        stats = self.mezo_pseudo_optimizer.last_debug_stats
        if stats is None:
            return {}
        return {
            "debug/projected_grad_before_update": stats["projected_grad"],
            "debug/loss_diff_before_update": stats["loss_diff"],
        }

    def _log_mezo_debug_metrics(self, logs: dict[str, float | str]) -> None:
        """Logs MeZO debug values into Trainer history."""
        if not logs:
            return
        self.log(logs)

    def _write_mezo_debug_rows(
        self,
        epoch: int,
        learning_rate: float,
        restore_logs: dict[str, float],
        update_logs: dict[str, float],
    ) -> None:
        """Writes MeZO debug summaries and parameter diffs as CSV artifacts."""
        debug_dir = Path(str(self.args.output_dir)).parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        self._write_mezo_debug_step_row(
            output_path=debug_dir / "mezo_debug_steps.csv",
            epoch=epoch,
            learning_rate=learning_rate,
            update_logs=update_logs,
        )
        self._write_mezo_debug_param_rows(
            output_path=debug_dir / "mezo_debug_params.csv",
            restore_logs=restore_logs,
            update_logs=update_logs,
        )

    def _write_mezo_debug_step_row(
        self,
        output_path: Path,
        epoch: int,
        learning_rate: float,
        update_logs: dict[str, float],
    ) -> None:
        """Appends one row with step-level MeZO debug values."""
        stats = self.mezo_pseudo_optimizer.last_debug_stats or {}
        row = {
            "step": self.state.global_step,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "param_dtype": str(next(self.model.parameters()).dtype),
            "zo_eps": self.mezo_pseudo_optimizer.zo_eps,
            "loss_plus": stats.get("loss_plus"),
            "loss_minus": stats.get("loss_minus"),
            "loss_diff": stats.get("loss_diff"),
            "projected_grad": stats.get("projected_grad"),
            "loss_before_update": update_logs.get("debug/loss_before_update"),
            "loss_after_update": update_logs.get("debug/loss_after_update"),
            "loss_after_minus_before": update_logs.get("debug/loss_after_minus_before"),
        }
        self._append_csv_row(output_path=output_path, row=row)

    def _write_mezo_debug_param_rows(
        self,
        output_path: Path,
        restore_logs: dict[str, float],
        update_logs: dict[str, float],
    ) -> None:
        """Appends one row per watched parameter with restore and update diffs."""
        restore_prefix = "debug/restore_max_diff/"
        update_prefix = "debug/update_max_diff/"
        param_names = sorted(
            {
                key.removeprefix(restore_prefix)
                for key in restore_logs
                if key.startswith(restore_prefix)
            }
            | {
                key.removeprefix(update_prefix)
                for key in update_logs
                if key.startswith(update_prefix)
            }
        )
        for param_name in param_names:
            row = {
                "step": self.state.global_step,
                "parameter_name": param_name,
                "restore_max_diff": restore_logs.get(f"{restore_prefix}{param_name}"),
                "update_max_diff": update_logs.get(f"{update_prefix}{param_name}"),
            }
            self._append_csv_row(output_path=output_path, row=row)

    def _append_csv_row(self, output_path: Path, row: dict[str, object]) -> None:
        """Appends a dictionary row to a CSV file, writing the header when needed."""
        file_exists = output_path.exists()
        with output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
