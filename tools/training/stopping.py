from __future__ import annotations

import math

import torch


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker

        self.patience  = self.config.early_stopping.patience
        self.min_delta = self.config.early_stopping.min_delta

        self.logger.section("[Early Stopping]")
        self.logger.kv_table({
            "Patience"  : self.patience,
            "Min Delta" : self.min_delta,
        })

        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss  = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter    = 0
            stop            = False

        else:
            self.counter += 1
            stop          = (self.counter >= self.patience)

        self.tracker.log_scalar("controls/early_stop_counter", self.counter, epoch)

        if stop:
            self.triggered = True
            self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {self.best_epoch + 1}.")

        return stop

    def reset(self) -> None:
        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False

    def state_dict(self) -> dict:
        return {
            "best_loss"  : self.best_loss,
            "counter"    : self.counter,
            "best_epoch" : self.best_epoch,
            "triggered"  : self.triggered,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_loss  = state["best_loss"]
        self.counter    = int(state["counter"])
        self.best_epoch = int(state["best_epoch"])
        self.triggered  = bool(state["triggered"])


class OverfitManager:
    def __init__(self, config, logger):
        self.enabled        = config.overfit.enabled
        self.max_steps      = config.overfit.max_steps
        self.stop_threshold = config.overfit.stop_threshold
        self.batch_size     = config.overfit.batch_size
        self.logger         = logger

        self.logger.section("[Overfit Manager]")
        self.logger.kv_table({
            "Enabled":     self.enabled,
            "Max Steps":   self.max_steps,
            "Stop Thresh": self.stop_threshold,
            "Batch Size":  self.batch_size,
        })

    def setup_loaders(self, train_loader, val_loader, test_loader):
        if not self.enabled:
            return train_loader, val_loader, test_loader

        raw_batch    = next(iter(train_loader))
        single_batch = tuple(t[:self.batch_size] if isinstance(t, torch.Tensor) else t for t in raw_batch)

        epoch_steps       = min(len(train_loader), self.max_steps)
        self._epoch_steps = epoch_steps
        self._steps_done  = 0

        data_loader     = [single_batch] * epoch_steps
        val_loader_out  = [single_batch]
        test_loader_out = [single_batch]

        return data_loader, val_loader_out, test_loader_out

    def planned_epochs(self) -> int:
        return math.ceil(self.max_steps / max(1, self._epoch_steps))

    def check_stop(self, train_loss: float) -> bool:
        if not self.enabled:
            return False

        self._steps_done += self._epoch_steps
        if self._steps_done >= self.max_steps:
            self.logger.warning(f"Overfit max_steps={self.max_steps} reached. Stopping.")
            return True

        if train_loss < self.stop_threshold:
            self.logger.warning(f"Training loss reached ~0 (loss={train_loss:.6f}). Stopping early.")
            return True

        return False
