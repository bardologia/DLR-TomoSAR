import torch


class OverfitManager:
    def __init__(self, config, logger):
        self.enabled = config.training.overfit_enabled
        self.logger  = logger
        
    def setup_loaders(self, train_loader, val_loader, test_loader):
        if not self.enabled:
            return train_loader, train_loader, val_loader, test_loader
            
        raw_batch         = next(iter(train_loader))
        single_batch      = tuple(t[:1] if isinstance(t, torch.Tensor) else t for t in raw_batch)
        overfit_steps     = min(len(train_loader), 50)
        
        data_loader       = [single_batch] * overfit_steps
        eval_train_loader = [single_batch]
        val_loader_out    = [single_batch]
        test_loader_out   = [single_batch]
        
        self.logger.warning(f"Overfitting mode enabled: training & validating on a subset of 1 inputs (repeated {overfit_steps} times for training).")
        return data_loader, eval_train_loader, val_loader_out, test_loader_out

    def check_stop(self, train_loss):
        if not self.enabled:
            return False
            
        if train_loss < 1e-6:
            self.logger.warning(f"Training loss reached ~0 (loss={train_loss:.6f}). Stopping early.")
            return True
        return False
