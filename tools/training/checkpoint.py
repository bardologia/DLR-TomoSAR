from __future__ import annotations

import os

import torch


class Checkpoint:
    def __init__(self, logger, tracker, save_path: str):
        self.logger          = logger
        self.tracker         = tracker
        self.save_path       = save_path
        self.best_val_loss   = float("inf")
        self.best_epoch      = -1
        self.loss_generation = 0

    def reset_baseline(self, loss_generation: int, epoch: int) -> None:
        self.best_val_loss   = float("inf")
        self.best_epoch      = -1
        self.loss_generation = int(loss_generation)
        self.logger.warning(f"Checkpoint baseline reset at epoch {epoch}: loss composition changed (generation {loss_generation}); best_val_loss is not comparable across the swap and tracking restarts on the new scale.")

    def step(self, val_loss: float, epoch: int, trainer) -> bool:
        loss_generation = int(trainer.criterion.loss_generation)
        if loss_generation != self.loss_generation:
            self.reset_baseline(loss_generation, epoch)

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

    def restore_best(self, model, device) -> None:
        if self.best_epoch < 0:
            self.logger.warning("No improving checkpoint was saved; leaving the model in its final-epoch state.")
            return

        checkpoint = torch.load(self.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["params"])

        self.logger.subsection(f"Restored best parameters from epoch {self.best_epoch + 1} (val_loss={self.best_val_loss:.4f}).")
