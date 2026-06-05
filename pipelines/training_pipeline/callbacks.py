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

    def state_dict(self) -> dict:
        return {
            "current_step"    : self.current_step,
            "warmup_finished" : self.warmup_finished,
            "mode"            : self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_step    = state["current_step"]
        self.warmup_finished = state["warmup_finished"]
        self.mode            = state.get("mode", self.mode)


class Scheduler:
    def __init__(self, base_lrs, warmup, config, logger, tracker):
        self.config         = config
        self.warmup         = warmup
        self.logger         = logger
        self.tracker        = tracker

        self.base_lrs       = list(base_lrs)
        self.current_lrs    = list(base_lrs)
        self.scheduler_type = self.config.scheduler.type

        self.last_metric    = None
        self.plateau_best   = float("inf")
        self.plateau_count  = 0
        self._epoch_offset  = 0

        self._log_scheduler_info()

    def _cosine_annealing(self, epoch: int) -> float:
        T_max         = self.config.scheduler.epochs
        eta_min       = float(self.config.scheduler.eta_min)
        base_lr       = self.base_lrs[0]
        eta_min_ratio = eta_min / max(base_lr, 1e-12)
        progress      = min(1.0, epoch / max(1, T_max))
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

    def _cosine_annealing_warm_restarts(self, epoch: int) -> float:
        T_0           = int(self.config.scheduler.T_0)
        T_mult        = float(self.config.scheduler.T_mult)
        eta_min       = float(self.config.scheduler.eta_min)
        base_lr       = self.base_lrs[0]
        eta_min_ratio = eta_min / max(base_lr, 1e-12)

        if T_mult == 1.0:
            T_cur = epoch % T_0
            T_i   = T_0
        else:
            n = math.floor(math.log(1 - epoch / T_0 * (1 - T_mult), T_mult)) if epoch >= T_0 else 0
            T_i   = T_0 * (T_mult ** n)
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)

        progress = T_cur / max(1, T_i)
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

    def _step_decay(self, epoch: int) -> float:
        step_size = int(self.config.scheduler.step_size)
        gamma     = float(self.config.scheduler.gamma)
        return gamma ** (epoch // step_size)

    def _multi_step(self, epoch: int) -> float:
        milestones = list(self.config.scheduler.milestones)
        gamma      = float(self.config.scheduler.gamma)
        n          = sum(1 for m in milestones if epoch >= m)
        return gamma ** n

    def _exponential(self, epoch: int) -> float:
        gamma = float(self.config.scheduler.gamma)
        return gamma ** epoch

    def _linear(self, epoch: int) -> float:
        start_factor = float(self.config.scheduler.start_factor)
        end_factor   = float(self.config.scheduler.end_factor)
        total_iters  = int(self.config.scheduler.total_iters)
        progress     = min(1.0, epoch / max(1, total_iters))
        return start_factor + (end_factor - start_factor) * progress

    def _polynomial(self, epoch: int) -> float:
        total_iters = int(self.config.scheduler.total_iters)
        power       = float(self.config.scheduler.power)
        progress    = min(1.0, epoch / max(1, total_iters))
        return (1.0 - progress) ** power

    def _constant(self) -> float:
        return 1.0

    def _reduce_on_plateau(self, metric: float | None) -> float:
        if metric is None:
            return 1.0

        factor    = float(self.config.scheduler.factor)
        patience  = int(self.config.scheduler.patience)
        threshold = float(self.config.scheduler.threshold)

        if metric < self.plateau_best - threshold:
            self.plateau_best  = metric
            self.plateau_count = 0
            return 1.0

        self.plateau_count += 1
        if self.plateau_count >= patience:
            self.plateau_count = 0
            self.plateau_best  = metric
            return factor

        return 1.0

    def _factor_for(self, epoch: int, metric: float | None) -> float:

        if self.scheduler_type == "cosine_annealing":
            return self._cosine_annealing(epoch)

        if self.scheduler_type == "cosine_annealing_warm_restarts":
            return self._cosine_annealing_warm_restarts(epoch)

        if self.scheduler_type == "step":
            return self._step_decay(epoch)

        if self.scheduler_type == "multi_step":
            return self._multi_step(epoch)

        if self.scheduler_type == "exponential":
            return self._exponential(epoch)

        if self.scheduler_type == "linear":
            return self._linear(epoch)

        if self.scheduler_type == "polynomial":
            return self._polynomial(epoch)

        if self.scheduler_type == "reduce_on_plateau":
            return self._reduce_on_plateau(metric)

        if self.scheduler_type == "constant":
            return self._constant()

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def reset(self, epoch_offset: int = 0) -> None:
        self._epoch_offset = epoch_offset
        self.current_lrs   = list(self.base_lrs)
        self.plateau_best  = float("inf")
        self.plateau_count = 0

    def step(self, epoch: int, metric: float | None = None) -> list[float]:
        factor           = self._factor_for(epoch - self._epoch_offset, metric)
        self.current_lrs = [lr * factor for lr in self.current_lrs] if self.scheduler_type == "reduce_on_plateau" else [lr * factor for lr in self.base_lrs]
        lrs              = list(self.current_lrs)

        if self.warmup is not None and not self.warmup.is_finished():
            f   = self.warmup.factor()
            lrs = [lr * f for lr in lrs]

        return lrs

    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        info = {
            "Scheduler Type" : self.scheduler_type,
            "Base LRs"       : self.base_lrs,
        }

        if self.scheduler_type == "cosine_annealing":
            info["T_max"]   = self.config.scheduler.epochs
            info["Eta Min"] = self.config.scheduler.eta_min

        if self.scheduler_type == "cosine_annealing_warm_restarts":
            info["T_0"]     = self.config.scheduler.T_0
            info["T_mult"]  = self.config.scheduler.T_mult
            info["Eta Min"] = self.config.scheduler.eta_min

        info["Warmup Enabled"] = self.warmup.enabled if self.warmup else False
        self.logger.kv_table(info)

    def state_dict(self) -> dict:
        return {
            "current_lrs"   : self.current_lrs,
            "plateau_best"  : self.plateau_best,
            "plateau_count" : self.plateau_count,
            "epoch_offset"  : self._epoch_offset,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_lrs   = state["current_lrs"]
        self.plateau_best  = state["plateau_best"]
        self.plateau_count = state["plateau_count"]
        self._epoch_offset = state.get("epoch_offset", 0)


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

    def state_dict(self) -> dict:
        return {
            "best_loss"   : self.best_loss,
            "counter"     : self.counter,
            "best_params" : self.best_params,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_loss   = state["best_loss"]
        self.counter     = state["counter"]
        self.best_params = state["best_params"]

    def reset(self) -> None:
        self.best_loss   = None
        self.counter     = 0
        self.best_epoch  = -1
        self.best_params = None
        self.triggered   = False

    def restore(self, model: torch.nn.Module):
        if self.restore_best and self.best_params is not None:
            model.load_state_dict(self.best_params)


class EMA:
    def __init__(self, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker
        self.enabled = self.config.ema.use_ema
        self.decay   = self.config.ema.ema_decay

        self.logger.section("[Exponential Moving Average (EMA)]")
        self.logger.kv_table({
            "Enabled": self.enabled,
            "Decay":   self.decay,
        })

        self.shadow = None
        self.backup = None

    def init(self, model: torch.nn.Module):
        if self.enabled:
            self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

        return self.shadow

    def update(self, model: torch.nn.Module, step: int = None):
        if not self.enabled or self.shadow is None:
            return None

        with torch.no_grad():
            names   = [name for name, param in model.named_parameters() if param.requires_grad]
            params  = [param for name, param in model.named_parameters() if param.requires_grad]
            shadows = [self.shadow[name] for name in names]

            torch._foreach_mul_(shadows, self.decay)
            torch._foreach_add_(shadows, params, alpha=1.0 - self.decay)

        if step is not None and self.tracker.debug:
            divergence = sum(torch.norm(self.shadow[name] - param).item() for name, param in model.named_parameters() if param.requires_grad)
            self.tracker.log_scalar("debug/ema_divergence", divergence, step)

        return self.shadow

    def apply_to(self, model: torch.nn.Module):
        if not self.enabled or self.shadow is None:
            return None

        self.backup = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        if not self.enabled or self.backup is None:
            return None

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])

        self.backup = None

    def state_dict(self) -> dict:
        return {
            "enabled" : self.enabled,
            "decay"   : self.decay,
            "shadow"  : self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled = state["enabled"]
        self.decay   = state["decay"]
        self.shadow  = state["shadow"]
        self.backup  = None


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
