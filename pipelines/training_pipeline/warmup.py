from __future__ import annotations
import math


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
        self.logger.subsection(f"Warmup Enabled      : {self.enabled}")
        self.logger.subsection(f"Warmup Steps        : {self.warmup_steps}")
        self.logger.subsection(f"Warmup Mode         : {self.mode}")
        self.logger.subsection(f"Warmup Start Factor : {self.warmup_start_factor} \n")

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
