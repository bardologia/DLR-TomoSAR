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

        self._log_scheduler_info()

    def _cosine_annealing(self, epoch: int) -> float:
        T_max    = self.config.scheduler.epochs
        eta_min  = float(self.config.scheduler.eta_min)
        progress = min(1.0, epoch / max(1, T_max))
        return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))

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
            return factor
       
        return 1.0

    def _factor_for(self, epoch: int, metric: float | None) -> float:
        
        if self.scheduler_type == "cosine_annealing":
            return self._cosine_annealing(epoch)
        
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
            f = self._reduce_on_plateau(metric)
            self.current_lrs = [lr * f for lr in self.current_lrs]
            return 1.0
        
        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def step(self, epoch: int, metric: float | None = None) -> list[float]:
        if self.warmup is not None and not self.warmup.is_finished():
            return [lr * self.warmup.factor() for lr in self.base_lrs]

        if self.scheduler_type == "reduce_on_plateau":
            self._reduce_on_plateau(metric)
            lrs = list(self.current_lrs)
        
        else:
            factor = self._factor_for(epoch, metric)
            lrs    = [lr * factor for lr in self.base_lrs]
            self.current_lrs = list(lrs)

        for i, lr in enumerate(lrs):
            self.tracker.log_scalar(f"lr/group_{i}", lr, epoch)
        
        return lrs

    def lrs_with_warmup(self, lrs: list[float]) -> list[float]:
        if self.warmup is None or self.warmup.is_finished():
            return lrs
        f = self.warmup.factor()
        
        return [lr * f for lr in lrs]

    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        self.logger.subsection(f"Scheduler Type    : {self.scheduler_type}")
        self.logger.subsection(f"Base LRs          : {self.base_lrs}")
        
        if self.scheduler_type == "cosine_annealing":
            self.logger.subsection(f"T_max             : {self.config.scheduler.epochs}")
            self.logger.subsection(f"Eta Min           : {self.config.scheduler.eta_min}")
        
        self.logger.subsection(f"Warmup Enabled    : {self.warmup.enabled if self.warmup else False} \n")

    def state_dict(self) -> dict:
        return {
            "current_lrs"   : self.current_lrs,
            "plateau_best"  : self.plateau_best,
            "plateau_count" : self.plateau_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_lrs   = state["current_lrs"]
        self.plateau_best  = state["plateau_best"]
        self.plateau_count = state["plateau_count"]
