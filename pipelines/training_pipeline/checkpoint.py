from __future__ import annotations

import os
import torch


class Checkpoint:
    def __init__(self, logger, tracker, save_path: str):
        self.logger        = logger
        self.tracker       = tracker
        self.save_path     = save_path
        self.best_val_loss = float("inf")
        self.best_epoch    = -1

    def step(self, val_loss: float, epoch: int, trainer) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.best_epoch    = int(epoch)
            self.save(trainer, self.save_path, epoch)
            self.logger.subsection(f"Checkpoint : new best  val_loss={self.best_val_loss:.4f}  -> {self.save_path}")
            return True
        else:
            self.logger.subsection(f"Checkpoint : no improvement  (best={self.best_val_loss:.4f} @ epoch {self.best_epoch})")
            return False

    def save(self, trainer, path: str, epoch: int) -> None:
        checkpoint = trainer.capture_state(epoch)
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_epoch"]    = self.best_epoch

        self.logger.info(f"Saving checkpoint at epoch {epoch} to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load(self, trainer, path: str) -> int:
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, weights_only=False)

        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch    = checkpoint["best_epoch"]

        epoch = trainer.restore_state(checkpoint)
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {epoch}")

        return epoch
