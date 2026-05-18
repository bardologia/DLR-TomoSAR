from __future__ import annotations

import torch
import numpy as np


class GradientClipper:
    def __init__(self, config, logger, tracker):
        self.logger      = logger
        self.tracker     = tracker       
        
        cfg              = config.gradient_clipper
        self.mode        = cfg.clip_mode
        self.threshold   = cfg.max_grad_norm if self.mode == "fixed" else None
        self.window      = cfg.adaptive_window
        self.percentile  = cfg.adaptive_percentile
        self.mean_std_k  = cfg.adaptive_mean_std_k
        self.epsilon     = cfg.clip_epsilon
        self.hist_freq   = cfg.log_histogram_freq
       
        self.history     : list[float] = []

        self.logger.section("[Gradient Clipper]")
        self.logger.subsection(f"Mode               : {self.mode}")
        
        if self.mode == "fixed":
            self.logger.subsection(f"Threshold          : {self.threshold}")
        
        elif self.mode == "adaptive_percentile":
            self.logger.subsection(f"Window             : {self.window}")
            self.logger.subsection(f"Percentile         : {self.percentile}")
        
        elif self.mode == "adaptive_mean_std":
            self.logger.subsection(f"Window             : {self.window}")
            self.logger.subsection(f"Mean+k*Std  k      : {self.mean_std_k}")
        
        self.logger.subsection("")

    @staticmethod
    def global_norm(model: torch.nn.Module) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _clip(self, model: torch.nn.Module, max_norm: float):
        norm = GradientClipper.global_norm(model)
        scale = min(1.0, max_norm / (norm + self.epsilon))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)
        return norm

    def _compute_adaptive_threshold(self) -> float | None:
        if len(self.history) < self.window:
            return None
        
        window_data = np.asarray(self.history[-self.window:], dtype=np.float32)
        
        if self.mode == "adaptive_percentile":
            return float(np.percentile(window_data, self.percentile))
        
        else:  
            return float(window_data.mean() + self.mean_std_k * window_data.std())

    def maybe_clip(self, model: torch.nn.Module, global_step: int):
        norm = GradientClipper.global_norm(model)

        if self.mode == "disabled":
            return norm

        if self.mode == "fixed":
            threshold = self.threshold
        else: 
            threshold = self._compute_adaptive_threshold()

        if threshold is None:
            return norm

        norm = self._clip(model, threshold)
        self.tracker.log_scalar("train/grad_clip_threshold", threshold, global_step)
        
        return norm

    def record(self, grad_norm_value: float, global_step: int):
        self.history.append(float(grad_norm_value))
        
        if global_step % self.hist_freq == 0 and len(self.history) >= self.hist_freq:
            self.tracker.log_histogram("train/grad_norm_dist", np.asarray(self.history[-self.hist_freq:], dtype=np.float32), global_step)
