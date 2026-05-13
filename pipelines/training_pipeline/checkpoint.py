import os
import torch


class Checkpoint:
    def __init__(self, logger, tracker, save_path: str):
        self.logger        = logger
        self.tracker       = tracker
        self.save_path     = save_path
        self.best_val_loss = float("inf")
        self.best_epoch    = -1
        self.best_metrics  = {}

    def step(self, val_loss: float, epoch: int, metrics: dict, trainer) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.best_epoch    = int(epoch)
            self.best_metrics  = dict(metrics)
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
            "best_metrics"  : self.best_metrics,
            "train_losses"  : trainer.train_losses,
            "val_losses"    : trainer.val_losses,
            
            "model_state_dict"     : trainer.model.state_dict(),
            "optimizer_state_dict" : trainer.optimizer.state_dict(),
            
            "config" : getattr(trainer.config, "to_dict", lambda: None)() or str(trainer.config),
            "x_axis" : trainer.x_axis.cpu(),

            "lr_scheduler_state_dict" : trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
            "ema_state_dict"          : trainer.ema.state_dict(),
            
            "early_stopping_state": {
                "best_loss"        : trainer.early_stopping.best_loss,
                "counter"          : trainer.early_stopping.counter,
                "best_model_state" : trainer.early_stopping.best_model_state,
            },
            
            "warmup_state": {
                "current_step"    : trainer.warmup.current_step,
                "warmup_finished" : trainer.warmup.warmup_finished,
            },
            
            "scaler_state_dict": trainer.scaler.state_dict() if trainer.scaler else None,
        }

        self.logger.info(f"Saving checkpoint at epoch {epoch} to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    def load(self, trainer, path: str) -> int:
    
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
        
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        trainer.global_step   = checkpoint["global_step"]
        self.best_val_loss    = checkpoint["best_val_loss"]
        self.best_epoch       = checkpoint["best_epoch"]
        self.best_metrics     = checkpoint["best_metrics"]
        trainer.train_losses  = checkpoint["train_losses"]
        trainer.val_losses    = checkpoint["val_losses"]
        
        if checkpoint["lr_scheduler_state_dict"] and trainer.lr_scheduler:
            trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        if checkpoint["ema_state_dict"]:
            trainer.ema.load_state_dict(checkpoint["ema_state_dict"])
        
        if "early_stopping_state" in checkpoint:
            es_state = checkpoint["early_stopping_state"]
            trainer.early_stopping.best_loss        = es_state["best_loss"]
            trainer.early_stopping.counter          = es_state["counter"]
            trainer.early_stopping.best_model_state = es_state["best_model_state"]
        
        if "warmup_state" in checkpoint:
            warmup_state = checkpoint["warmup_state"]
            trainer.warmup.current_step    = warmup_state["current_step"]
            trainer.warmup.warmup_finished = warmup_state["warmup_finished"]
        
        if checkpoint["scaler_state_dict"] and trainer.scaler:
            trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        epoch = checkpoint["epoch"]
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {epoch}")
        return epoch
