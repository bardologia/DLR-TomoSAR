import torch


class GradientClipper:
    def __init__(self, initial_threshold: float | None, logger, tracker):
        self.threshold = initial_threshold
        self.logger    = logger
        self.tracker   = tracker
        self.history   = []
        
        self.logger.section("[Gradient Clipper]")
        if self.threshold is None:
            self.logger.subsection(f"Mode               : Disabled (logging norms only)")
        else:
            self.logger.subsection(f"Threshold          : {self.threshold}")
            self.logger.subsection(f"Mode               : Enabled")
        self.logger.subsection("")

    def step(self, parameters, global_step: int) -> float:
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in parameters if p.grad is not None) ** 0.5
        
        if self.threshold is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.threshold)

        self.history.append(total_norm)
        
        if global_step % 100 == 0 and len(self.history) >= 100:
            self.tracker.log_histogram("train/grad_norm_dist", torch.tensor(self.history[-100:]), global_step)
                
        return total_norm
