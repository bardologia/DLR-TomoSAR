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


class OverfitManager:
    def __init__(self, config, logger):
        self.enabled         = config.overfit.enabled
        self.max_steps       = config.overfit.max_steps
        self.stop_threshold  = config.overfit.stop_threshold
        self.batch_size      = config.overfit.batch_size
        self.logger          = logger

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

        data_loader       = [single_batch] * epoch_steps
        val_loader_out    = [single_batch]
        test_loader_out   = [single_batch]

        return data_loader, val_loader_out, test_loader_out

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


class CurriculumController:
    def __init__(self, curriculum, criterion, early_stopping, lr_scheduler, warmup, optimizer, update_optimizer, logger):
        self.curriculum        = curriculum
        self.criterion         = criterion
        self.early_stopping    = early_stopping
        self.lr_scheduler      = lr_scheduler
        self.warmup            = warmup
        self.optimizer         = optimizer
        self.update_optimizer  = update_optimizer
        self.logger            = logger

    def maybe_swap(self, epoch: int) -> bool:
        lc = self.curriculum
        if not (lc.enabled and epoch == lc.swap_epoch):
            return False

        self.logger.section(f"[Curriculum Loss Swap @ epoch {epoch + 1}]")
        self.criterion.set_curriculum(lc.complete)
        self.logger.subsection("Loss config replaced with curriculum.loss.complete.")
        self.logger.subsection(f"ParamMatcher strategy updated to '{lc.complete.param_match}'.")

        if lc.reset_early_stopping:
            self.early_stopping.reset()
            self.logger.subsection("Early stopping reset.")

        if lc.reset_lr:
            self.lr_scheduler.reset(epoch_offset=epoch)
            self.logger.subsection(f"LR scheduler reset (epoch offset = {epoch}).")

        if lc.reset_warmup:
            self.warmup.reset()
            self.logger.subsection(f"Warmup reset ({self.warmup.warmup_steps} steps).")

        if lc.reset_optimizer:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    state = self.optimizer.state.get(p)
                    if state:
                        state.clear()

            self.logger.subsection("Optimizer state (Adam moments) cleared.")

        if lc.reset_lr or lc.reset_warmup:
            warmup_factor = self.warmup.factor() if (self.warmup.enabled and not self.warmup.is_finished()) else 1.0
            immediate_lrs = [lr * warmup_factor for lr in self.lr_scheduler.base_lrs]
            self.update_optimizer(immediate_lrs)
            self.logger.subsection(f"Optimizer LR set to warmup-adjusted value (factor={warmup_factor:.4f}) for swap epoch.")

        return True
