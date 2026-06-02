from __future__ import annotations

import torch


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
