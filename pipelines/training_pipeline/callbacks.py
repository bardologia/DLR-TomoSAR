from __future__ import annotations

import math

import numpy as np
import torch


class Warmup:
    def __init__(self, config, logger, tracker):
        self.config              = config
        self.logger              = logger
        self.tracker             = tracker

        self.warmup_steps        = self.config.warmup.warmup_steps
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.mode                = self.config.warmup.warmup_mode
        self.poly_power          = self.config.warmup.warmup_poly_power

        self.current_step        = 0
        self.warmup_finished     = False
        self._logged_completion  = False

        self.logger.section("[Warmup Scheduler]")
        self.logger.kv_table({
            "Enabled"      : self.enabled,
            "Steps"        : self.warmup_steps,
            "Mode"         : self.mode,
            "Start Factor" : self.warmup_start_factor,
        })

    def factor(self) -> float:
        if not self.enabled or self.warmup_steps <= 0:
            return 1.0

        if self.current_step >= self.warmup_steps:
            return 1.0

        progress = self.current_step / self.warmup_steps
        s        = self.warmup_start_factor

        if self.mode == "cosine":
            cos_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
            return s + (1.0 - s) * cos_factor

        elif self.mode == "exponential":
            if s <= 0:
                return progress
            return s ** (1.0 - progress)

        elif self.mode == "polynomial":
            return s + (1.0 - s) * (progress ** self.poly_power)

        else:
            return s + (1.0 - s) * progress

    def step(self) -> float:
        if not self.enabled or self.warmup_steps <= 0:
            self.warmup_finished = True
            return 1.0

        if self.warmup_finished:
            return 1.0

        self.current_step += 1
        factor             = self.factor()
        self.tracker.log_scalar("lr/warmup_factor", factor, self.current_step)

        if self.current_step >= self.warmup_steps and not self.warmup_finished:
            self.warmup_finished = True
            if not self._logged_completion:
                self.logger.info(f"Warmup completed at step {self.current_step}.")
                self._logged_completion = True

        return factor

    def reset(self) -> None:
        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0


class Scheduler:
    def __init__(self, base_lrs, warmup, config, logger, tracker):
        self.config         = config
        self.warmup         = warmup
        self.logger         = logger
        self.tracker        = tracker

        self.base_lrs       = list(base_lrs)
        self.current_lrs    = list(base_lrs)
        self.scheduler_type = self.config.scheduler.type

        self._epoch_offset  = 0

        self._log_scheduler_info()

    def _cosine_annealing(self, epoch: int) -> float:
        T_max         = self.config.scheduler.epochs
        eta_min       = float(self.config.scheduler.eta_min)
        base_lr       = self.base_lrs[0]
        eta_min_ratio = eta_min / max(base_lr, 1e-12)
        progress      = min(1.0, epoch / max(1, T_max))
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

    def _constant(self) -> float:
        return 1.0

    def _factor_for(self, epoch: int) -> float:
        if self.scheduler_type == "cosine_annealing":
            return self._cosine_annealing(epoch)

        if self.scheduler_type == "constant":
            return self._constant()

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def reset(self, epoch_offset: int = 0) -> None:
        self._epoch_offset = epoch_offset
        self.current_lrs   = list(self.base_lrs)

    def step(self, epoch: int) -> list[float]:
        factor           = self._factor_for(epoch - self._epoch_offset)
        self.current_lrs = [lr * factor for lr in self.base_lrs]

        return list(self.current_lrs)

    def effective_lrs(self) -> list[float]:
        if self.warmup is not None and not self.warmup.is_finished():
            f = self.warmup.factor()
            return [lr * f for lr in self.current_lrs]

        return list(self.current_lrs)

    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        info = {
            "Scheduler Type" : self.scheduler_type,
            "Base LRs"       : self.base_lrs,
        }

        if self.scheduler_type == "cosine_annealing":
            info["T_max"]   = self.config.scheduler.epochs
            info["Eta Min"] = self.config.scheduler.eta_min

        info["Warmup Enabled"] = self.warmup.enabled if self.warmup else False
        self.logger.kv_table(info)


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config       = config
        self.logger       = logger
        self.tracker      = tracker

        self.patience     = self.config.early_stopping.patience
        self.min_delta    = self.config.early_stopping.min_delta
        self.restore_best = self.config.early_stopping.restore_best

        self.logger.section("[Early Stopping]")
        self.logger.kv_table({
            "Patience"     : self.patience,
            "Min Delta"    : self.min_delta,
            "Restore Best" : self.restore_best,
        })

        self.best_loss        = None
        self.counter          = 0
        self.best_epoch       = -1
        self.best_params      = None
        self.triggered        = False

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss   = float(val_loss)
            self.best_epoch  = int(epoch)
            self._save_state(model)
            self.counter     = 0
            stop             = False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss   = float(val_loss)
            self.best_epoch  = int(epoch)
            self.counter     = 0
            self._save_state(model)
            self.tracker.log_scalar("early_stop/best_val_loss", self.best_loss, epoch)
            stop = False

        else:
            self.counter += 1
            stop          = (self.counter >= self.patience)

        self.tracker.log_scalar("early_stop/counter", self.counter, epoch)

        if stop:
            self.triggered = True
            self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {self.best_epoch + 1}.")

        return stop

    def _save_state(self, model: torch.nn.Module):
        if self.restore_best:
            self.best_params = {name: param.clone().detach().cpu() for name, param in model.state_dict().items()}

    def reset(self) -> None:
        self.best_loss   = None
        self.counter     = 0
        self.best_epoch  = -1
        self.best_params = None
        self.triggered   = False

    def restore(self, model: torch.nn.Module):
        if self.restore_best and self.best_params is not None:
            model.load_state_dict(self.best_params)


class GradientClipper:
    def __init__(self, config, logger, tracker):
        self.logger      = logger
        self.tracker     = tracker

        self.mode        = config.gradient_clipper.clip_mode
        self.threshold   = config.gradient_clipper.max_grad_norm if self.mode == "fixed" else None
        self.window      = config.gradient_clipper.adaptive_window
        self.percentile  = config.gradient_clipper.adaptive_percentile
        self.mean_std_k  = config.gradient_clipper.adaptive_mean_std_k
        self.epsilon     = config.gradient_clipper.clip_epsilon
        self.hist_freq   = config.gradient_clipper.log_histogram_freq

        self.history     : list[float] = []

        self.logger.section("[Gradient Clipper]")
        self.logger.subsection(f"Mode : {self.mode}")

        if self.mode == "fixed":
            self.logger.subsection(f"Threshold     : {self.threshold}")

        elif self.mode == "adaptive_percentile":
            self.logger.subsection(f"Window        : {self.window}")
            self.logger.subsection(f"Percentile    : {self.percentile}")

        elif self.mode == "adaptive_mean_std":
            self.logger.subsection(f"Window        : {self.window}")
            self.logger.subsection(f"Mean+k*Std  k : {self.mean_std_k}")

    @staticmethod
    def global_norm(model: torch.nn.Module) -> float:
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

        if not grads:
            return 0.0

        per_param_norms = torch._foreach_norm(grads, 2)
        total_norm      = torch.norm(torch.stack(per_param_norms), 2)

        return total_norm.item()

    def _clip(self, model: torch.nn.Module, norm: float, max_norm: float) -> tuple[float, float]:
        scale = min(1.0, max_norm / (norm + self.epsilon))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)

        norm_after = norm * scale
        return norm, norm_after

    def _compute_adaptive_threshold(self) -> float | None:
        if len(self.history) < self.window:
            return None

        window_data = np.asarray(self.history[-self.window:], dtype=np.float32)

        if self.mode == "adaptive_percentile":
            return float(np.percentile(window_data, self.percentile))

        else:
            return float(window_data.mean() + self.mean_std_k * window_data.std())

    def check_gradients(self, model: torch.nn.Module, global_step: int) -> bool:
        has_invalid = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"NaN/Inf gradient detected in {name} at step {global_step}! GradScaler will handle this.")
                    has_invalid = True
                    break

        return has_invalid

    def maybe_clip(self, model: torch.nn.Module, global_step: int):
        if self.tracker.debug:
            self.check_gradients(model, global_step)

        norm = GradientClipper.global_norm(model)

        if norm > 100.0:
            self.logger.warning(f"Exploding gradient norm detected: {norm:.2f} at step {global_step}!")

        self.tracker.log_scalar("train/grad_norm_before_clip", norm, global_step)

        if self.mode == "disabled":
            return norm

        if self.mode == "fixed":
            threshold = self.threshold
        else:
            threshold = self._compute_adaptive_threshold()

        if threshold is None:
            return norm

        norm_before, norm_after = self._clip(model, norm, threshold)
        clip_ratio = norm_after / (norm_before + self.epsilon)

        self.tracker.log_scalar("train/grad_norm_after_clip",  norm_after,  global_step)
        self.tracker.log_scalar("train/grad_clip_ratio",       clip_ratio,  global_step)
        self.tracker.log_scalar("train/grad_clip_threshold",   threshold,   global_step)

        return norm_after

    def record(self, grad_norm_value: float, global_step: int):
        self.history.append(float(grad_norm_value))

        if global_step % self.hist_freq == 0 and len(self.history) >= self.hist_freq:
            self.tracker.log_histogram("train/grad_norm_dist", np.asarray(self.history[-self.hist_freq:], dtype=np.float32), global_step)
