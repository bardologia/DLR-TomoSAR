import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker
        self.enabled = self.config.ema.use_ema
        self.decay   = self.config.ema.ema_decay
        
        self.logger.section("[Exponential Moving Average (EMA)]")
        self.logger.subsection(f"EMA Enabled: {self.enabled}")
        self.logger.subsection(f"EMA Decay  : {self.decay} \n")

        self.shadow  = {}
        self.backup  = {}
        
        if self.enabled:
            self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int = None) -> None:
        if not self.enabled:
            return
        
        total_divergence = 0.0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                self.logger.warning(f"EMA: Initialized shadow for new parameter '{name}'")
                continue
            
            divergence        = (self.shadow[name] - param.detach()).norm().item()
            total_divergence += divergence
            
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        
        if step is not None and self.tracker.debug:
            self.tracker.log_scalar("debug/ema_divergence", total_divergence, step)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        
        self.backup = {}
        
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "enabled" : self.enabled,
            "decay"   : self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled = state['enabled']
        self.decay   = state['decay']
        self.shadow  = state['shadow']
        self.backup  = {}
