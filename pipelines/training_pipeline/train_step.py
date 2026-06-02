from __future__ import annotations

import torch


class TrainStep:
    def __init__(self, model, optimizer, scaler, criterion, grad_clipper, ema, device, logger, tracker, accumulation_steps, use_amp, ema_every):
        self.model              = model
        self.optimizer          = optimizer
        self.scaler             = scaler
        self.criterion          = criterion
        self.grad_clipper       = grad_clipper
        self.ema                = ema
        self.device             = device
        self.logger             = logger
        self.tracker            = tracker
        self.accumulation_steps = accumulation_steps
        self.use_amp            = use_amp
        self.ema_every          = ema_every

    def step(self, images: torch.Tensor, gt_params: torch.Tensor | None, batch_idx: int, n_batches: int, global_step: int):
        loss, loss_dict = self._forward(images, gt_params, global_step)
        self._backward(loss, batch_idx, n_batches, global_step)

        return loss, loss_dict

    def _forward(self, images: torch.Tensor, gt_params: torch.Tensor | None, global_step: int):
        if self.tracker.debug:
            if torch.isnan(images).any() or torch.isinf(images).any():
                self.logger.warning(f"NaN or Inf detected in input images at step {global_step}!")
            if gt_params is not None and (torch.isnan(gt_params).any() or torch.isinf(gt_params).any()):
                self.logger.warning(f"NaN or Inf detected in ground truth parameters at step {global_step}!")

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            pred_params = self.model(images)
            if self.tracker.debug and (torch.isnan(pred_params).any() or torch.isinf(pred_params).any()):
                self.logger.warning(f"NaN or Inf detected in model predictions at step {global_step}!")

            loss_dict   = self.criterion(pred_params, gt_params)
            loss        = loss_dict["total_loss"]
            loss        = loss / self.accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning(f"Total loss evaluated to NaN or Inf at step {global_step}!")

        return loss, loss_dict

    def _backward(self, loss: torch.Tensor, batch_idx: int, n_batches: int, global_step: int) -> None:
        self.scaler.scale(loss).backward()

        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            self.scaler.unscale_(self.optimizer)

            grad_norm = self.grad_clipper.maybe_clip(self.model, global_step)
            self.grad_clipper.record(grad_norm, global_step)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.ema.enabled:
                if global_step % self.ema_every == 0:
                    self.ema.update(self.model, step=global_step)
