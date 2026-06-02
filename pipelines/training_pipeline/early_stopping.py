from __future__ import annotations

import torch


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config       = config
        self.logger       = logger
        self.tracker      = tracker
        
        self.patience     = self.config.early_stopping.patience
        self.min_delta    = self.config.early_stopping.min_delta
        self.restore_best = self.config.early_stopping.restore_best

        self.logger.section("[Early Stopping]")
        self.logger.kv_table({
            "Patience"     : self.patience,
            "Min Delta"    : self.min_delta,
            "Restore Best" : self.restore_best,
        })

        self.best_loss        = None
        self.counter          = 0
        self.best_epoch       = -1
        self.best_params      = None
        self.triggered        = False

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss   = float(val_loss)
            self.best_epoch  = int(epoch)
            self._save_state(model)
            self.counter     = 0
            stop             = False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss   = float(val_loss)
            self.best_epoch  = int(epoch)
            self.counter     = 0
            self._save_state(model)
            self.tracker.log_scalar("early_stop/best_val_loss", self.best_loss, epoch)
            stop = False

        else:
            self.counter += 1
            stop          = (self.counter >= self.patience)

        self.tracker.log_scalar("early_stop/counter", self.counter, epoch)

        if stop:
            self.triggered = True
            self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {self.best_epoch + 1}.")

        return stop

    def _save_state(self, model: torch.nn.Module):
        if self.restore_best:
            self.best_params = {name: param.clone().detach().cpu() for name, param in model.state_dict().items()}

    def state_dict(self) -> dict:
        return {
            "best_loss"   : self.best_loss,
            "counter"     : self.counter,
            "best_params" : self.best_params,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_loss   = state["best_loss"]
        self.counter     = state["counter"]
        self.best_params = state["best_params"]

    def reset(self) -> None:
        self.best_loss   = None
        self.counter     = 0
        self.best_epoch  = -1
        self.best_params = None
        self.triggered   = False

    def restore(self, model: torch.nn.Module):
        if self.restore_best and self.best_params is not None:
            model.load_state_dict(self.best_params)
