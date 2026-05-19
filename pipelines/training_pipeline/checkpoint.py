from __future__ import annotations

import os
import torch
import numpy as np


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
        checkpoint = {
            "epoch"         : epoch,
            "global_step"   : trainer.global_step,
            "best_val_loss" : self.best_val_loss,
            "best_epoch"    : self.best_epoch,
            "train_losses"  : trainer.train_losses,
            "val_losses"    : trainer.val_losses,

            "params"        : trainer.model.state_dict(),
            "opt_state"     : trainer.optimizer.state_dict(),
            "batch_stats"   : getattr(trainer, "_batch_stats", None),
            "ema_shadow"    : trainer.ema.state_dict() if hasattr(trainer, "ema") and trainer.ema is not None else None,

            "config"        : getattr(trainer.config, "to_dict", lambda: None)() or str(trainer.config),
            "x_axis"        : trainer.x_axis.cpu().numpy() if torch.is_tensor(trainer.x_axis) else np.asarray(trainer.x_axis),

            "scheduler_state" : trainer.lr_scheduler.state_dict(),
            "warmup_state"    : trainer.warmup.state_dict(),

            "early_stopping_state": {
                "best_loss"   : trainer.early_stopping.best_loss,
                "counter"     : trainer.early_stopping.counter,
                "best_params" : trainer.early_stopping.best_params,
            },
        }

        self.logger.info(f"Saving checkpoint at epoch {epoch} to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load(self, trainer, path: str) -> int:
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, weights_only=False)

        trainer.model.load_state_dict(checkpoint["params"])
        trainer.optimizer.load_state_dict(checkpoint["opt_state"])
        if hasattr(trainer, "_batch_stats"):
            trainer._batch_stats = checkpoint.get("batch_stats")
            
        if checkpoint.get("ema_shadow") is not None and hasattr(trainer, "ema") and trainer.ema.enabled:
            trainer.ema.load_state_dict(checkpoint["ema_shadow"])

        trainer.global_step  = checkpoint["global_step"]
        self.best_val_loss   = checkpoint["best_val_loss"]
        self.best_epoch      = checkpoint["best_epoch"]
        trainer.train_losses = checkpoint["train_losses"]
        trainer.val_losses   = checkpoint["val_losses"]

        trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        trainer.warmup.load_state_dict(checkpoint["warmup_state"])

        es_state = checkpoint["early_stopping_state"]
        trainer.early_stopping.best_loss   = es_state["best_loss"]
        trainer.early_stopping.counter     = es_state["counter"]
        trainer.early_stopping.best_params = es_state["best_params"]

        epoch = checkpoint["epoch"]
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {epoch}")
        
        return epoch
