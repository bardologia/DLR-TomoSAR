from __future__ import annotations

import numpy as np


class OverfitManager:
    def __init__(self, config, logger):
        self.enabled         = config.training.overfit_enabled
        self.max_steps       = config.training.overfit_max_steps
        self.stop_threshold  = config.training.overfit_stop_threshold
        self.batch_size      = config.training.overfit_batch_size
        self.logger          = logger

        self.logger.section("[Overfit Manager]")
        self.logger.subsection(f"Overfit Enabled     : {self.enabled}")
        self.logger.subsection(f"Overfit Max Steps   : {self.max_steps}")
        self.logger.subsection(f"Overfit Stop Thresh : {self.stop_threshold}")
        self.logger.subsection(f"Overfit Batch Size  : {self.batch_size} \n")

    def setup_loaders(self, train_loader, val_loader, test_loader):
        if not self.enabled:
            return train_loader, train_loader, val_loader, test_loader

        raw_batch     = next(iter(train_loader))
        single_batch  = tuple(t[:self.batch_size] if isinstance(t, np.ndarray) else t for t in raw_batch)
        overfit_steps = min(len(train_loader), self.max_steps)

        data_loader       = [single_batch] * overfit_steps
        eval_train_loader = [single_batch]
        val_loader_out    = [single_batch]
        test_loader_out   = [single_batch]

        return data_loader, eval_train_loader, val_loader_out, test_loader_out

    def check_stop(self, train_loss: float) -> bool:
        if not self.enabled:
            return False
      
        if train_loss < self.stop_threshold:
            self.logger.warning(f"Training loss reached ~0 (loss={train_loss:.6f}). Stopping early.")
            return True
      
        return False
