
class Warmup:
    def __init__(self, optimizer, config, logger, tracker):
        self.config              = config
        self.logger              = logger
        self.tracker             = tracker
        self.optimizer           = optimizer
        self.warmup_steps        = self.config.warmup.warmup_steps
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.base_lrs            = [group['lr'] for group in optimizer.param_groups]
        self.current_step        = 0
        self.warmup_finished     = False
        self._logged_completion  = False

        self.logger.section("[Warmup Scheduler]")
        self.logger.subsection(f"Warmup Enabled      : {self.enabled}")
        self.logger.subsection(f"Warmup Steps        : {self.warmup_steps}")
        self.logger.subsection(f"Warmup Start Factor : {self.warmup_start_factor} \n")

        if self.enabled and self.warmup_steps > 0:
            self._apply_warmup_factor(self.warmup_start_factor)
    
    def _apply_warmup_factor(self, factor: float) -> None:
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor
        
    def step(self) -> None:
        if not self.enabled or self.warmup_steps <= 0:
            return

        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            self._apply_warmup_factor(factor)
            self.tracker.log_scalar("lr/warmup_factor", factor, self.current_step)
        
        elif not self.warmup_finished:
            self._apply_warmup_factor(1.0)
            self.warmup_finished = True
            
            if not self._logged_completion:
                self.logger.info(f"Warmup completed at step {self.current_step}. Learning rates restored to base values.")
                self.tracker.log_scalar("lr/warmup_factor", 1.0, self.current_step)
                self._logged_completion = True
    
    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0
