from __future__ import annotations

import math


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
        T_0       = int(self.config.scheduler.T_0)
        T_mult    = float(self.config.scheduler.T_mult)
        eta_min   = float(self.config.scheduler.eta_min)
        base_lr   = self.base_lrs[0]
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
            "Scheduler Type": self.scheduler_type,
            "Base LRs":       self.base_lrs,
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
