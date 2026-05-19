from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4" 
 
import gc
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tools                                          import Tracker, ResourceMonitor
from pipelines.training_pipeline.early_stopping     import EarlyStopping
from pipelines.training_pipeline.warmup             import Warmup
from pipelines.training_pipeline.scheduler          import Scheduler
from pipelines.training_pipeline.ema                import EMA
from pipelines.training_pipeline.gradient_clipper   import GradientClipper
from pipelines.training_pipeline.overfit            import OverfitManager
from pipelines.training_pipeline.checkpoint         import Checkpoint
from pipelines.training_pipeline.loss               import Loss


class Trainer:
    def __init__(self, model, x_axis, config, run_dir, logger, norm_stats=None):
        self.logger       = logger
        self.config       = config
        self.gaussian_cfg = config.gaussian
        self.loss_cfg     = config.loss
        self.norm_stats   = norm_stats

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.section("[Training Start]")
        self.logger.subsection(f"Backend       : PyTorch")
        self.logger.subsection(f"Device        : {self.device}")
        self.logger.subsection(f"Log Directory : {self.config.io.logdir} \n")

        self.checkpoint_path = Path(run_dir) / "best_model.pt"
        self.tracker         = Tracker(writer=self.config.io.writer, debug=getattr(self.config.training, "log_debug", False))

        self.model  = model.to(self.device)
        self.x_axis = torch.tensor(x_axis, device=self.device, dtype=torch.float32)

        self.epochs               = self.config.training.epochs
        self.validation_frequency = self.config.training.validation_frequency
        self.accumulation_steps   = self.config.training.gradient_accumulation_steps
        self.use_amp              = getattr(self.config.training, "use_amp", False)
        
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.base_lrs   = [float(getattr(self.config.optimizer, "lr", 1e-3))]
        self.optimizer  = self._build_optimizer(self.base_lrs)

        self.warmup           = Warmup(self.config, self.logger, self.tracker)
        self.lr_scheduler     = Scheduler(self.base_lrs, self.warmup, self.config, self.logger, self.tracker)
        self.ema              = EMA(self.config, self.logger, self.tracker)
        self.early_stopping   = EarlyStopping(self.config, self.logger, self.tracker)
        self.criterion        = Loss(self.x_axis, self.logger, self.tracker, self.gaussian_cfg, self.loss_cfg, norm_stats=self.norm_stats)
        self.checkpoint       = Checkpoint(self.logger, self.tracker, str(self.checkpoint_path))
        self.grad_clipper     = GradientClipper(config = self.config, logger = self.logger, tracker = self.tracker)
        self.overfitter       = OverfitManager(self.config, self.logger)
        self.resource_monitor = ResourceMonitor(config = self.config.resources, logger = self.logger, tracker = self.tracker, step_getter = lambda: self.global_step)
        
        self.ema.init(self.model)

        self.global_step  = 0
        self.train_losses = []
        self.val_losses   = []

    def _build_optimizer(self, base_lrs):
        lr           = base_lrs[0]
        betas        = tuple(self.config.optimizer.betas)
        eps          = self.config.optimizer.eps
        weight_decay = float(getattr(self.config.optimizer, "weight_decay", 0.0))

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        return optimizer

    def _update_optimizer(self, lrs: list[float]):
        lr = lrs[0]
        current_lr = getattr(self, "_current_lr", None)
        if current_lr is not None and abs(lr - current_lr) < 1e-12:
            return

        self._current_lr = lr
        self.tracker.log_scalar("lr/learning_rate", lr, self.global_step)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _forward(self, images: torch.Tensor, gt_params: torch.Tensor | None) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            pred_params = self.model(images)
            loss_dict   = self.criterion(pred_params, gt_params)
            loss        = loss_dict["total_loss"]
            loss        = loss / self.accumulation_steps

        return loss

    def _backward(self, loss: torch.Tensor, batch_idx: int, n_batches: int) -> None:
        self.scaler.scale(loss).backward()

        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.grad_clipper.maybe_clip(self.model, self.global_step)
            self.grad_clipper.record(grad_norm, self.global_step)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.ema.enabled:
                ema_every = max(1, int(getattr(self.config.ema, "update_every_n_steps", 10)))
                if self.global_step % ema_every == 0:
                    self.ema.update(self.model, step=self.global_step)

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        num_batches = 0
        loss_sum    = 0.0

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]Epoch {epoch+1}/{self.epochs}[/section] - train", total=len(train_loader))
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch[0].to(self.device)
                gt_params = batch[1].to(self.device) if len(batch) > 1 and batch[1] is not None else None

                loss = self._forward(images, gt_params)
                self._backward(loss, batch_idx, len(train_loader))
                
                loss_val = loss.item() * self.accumulation_steps
                loss_sum += loss_val
                num_batches += 1

                self.warmup.step()
                self.global_step += 1

                clear_n = self.config.memory.clear_cache_every_n_steps
                if clear_n > 0 and self.global_step % clear_n == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                _prog.update(_task, advance=1)

        avg_loss = loss_sum / num_batches if num_batches else float("nan")
        self.tracker.log_scalar("loss/train", avg_loss, epoch)
        self.tracker.log_memory(epoch)

        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage="validation"):
        mem_cfg = self.config.memory

        self.ema.apply_to(self.model)
        self.model.eval()

        try:
            total_loss = 0.0
            num_batches = 0

            with self.logger.track(transient=True) as _prog:
                _task = _prog.add_task(f"[section]Eval {stage}[/section] - epoch {epoch+1}/{self.epochs}", total=len(loader))
                for batch_idx, batch in enumerate(loader):
                    images = batch[0].to(self.device)
                    gt_params = batch[1].to(self.device) if len(batch) > 1 and batch[1] is not None else None
                    
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                        pred_params = self.model(images)
                        loss_dict   = self.criterion(pred_params, gt_params)
                        loss        = loss_dict["total_loss"]

                    total_loss += loss.item()
                    num_batches += 1
                    _prog.advance(_task)

            avg_loss = total_loss / max(1, num_batches)

        finally:
            self.ema.restore(self.model)
            if mem_cfg.clear_cache_after_eval:
                gc.collect()
                torch.cuda.empty_cache()

        return {
            "avg_loss"    : avg_loss,
            "num_batches" : num_batches,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.logger.section("[PyTorch Training Loop]")
        self.logger.subsection(f"Train loader size      = {len(train_loader)}")
        self.logger.subsection(f"Validation loader size = {len(val_loader)}")
        self.logger.subsection(f"Test loader size       = {len(test_loader)}")
        self.logger.subsection(f"Device                 = {self.device} \n")

        gc.collect()
        torch.cuda.empty_cache()

        if self.resource_monitor is not None:
            self.resource_monitor.start()

        try:
            data_loader, val_loader, test_loader = self.overfitter.setup_loaders(train_loader, val_loader, test_loader)

            epochs = self.epochs
            with self.logger.live_monitor("Training Progress") as live_mon:
                with self.logger.track(transient=False) as _prog_epochs:
                    _task_epochs = _prog_epochs.add_task("[section]Training[/section]", total=epochs)
                    for epoch in range(epochs):
                        epoch_num = epoch + 1

                        self.logger.section(f"[Epoch {epoch_num}/{epochs}]")
                        train_loss = self.train_epoch(data_loader, epoch)
                        self.logger.subsection(f"Train  : loss={train_loss:.4f}")

                        do_eval = (epoch_num % self.validation_frequency == 0) or (epoch_num == epochs)

                        if do_eval:
                            val_results = self.evaluate(val_loader, epoch, stage="validation")
                            val_loss    = val_results["avg_loss"]
                            self.logger.subsection(f"Validation : loss={val_loss:.4f}  (batches={val_results['num_batches']})")

                            self.tracker.log_scalar("loss/val",     val_loss,                 epoch)
                
                            self.checkpoint.step(val_loss, epoch_num, self)
                            new_lrs = self.lr_scheduler.step(epoch, metric=val_loss)
                            stop    = self.early_stopping(val_loss, self.model, epoch)
                        else:
                            val_results  = {"avg_loss": float("nan"), "num_batches": 0}
                            val_loss     = val_results["avg_loss"]

                            new_lrs = self.lr_scheduler.step(epoch, metric=None)
                            stop    = False

                        self.train_losses.append(train_loss)
                        self.val_losses.append(val_loss)

                        if self.config.memory.clear_cache_after_epoch:
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                        self._update_optimizer(self.lr_scheduler.lrs_with_warmup(new_lrs))

                        monitor_data = {
                            "epoch"         : f"{epoch_num}/{epochs}",
                            "train_loss"    : train_loss,
                            "val_loss"      : val_loss,
                            "best_val_loss" : self.checkpoint.best_val_loss,
                            "best_epoch"    : self.checkpoint.best_epoch,
                            "lr"            : new_lrs[0],
                        }
                        live_mon.update(**monitor_data)

                        _prog_epochs.update(_task_epochs, advance=1, description=f"[section]Training[/section]  best_val={self.checkpoint.best_val_loss:.4f} @ ep {self.checkpoint.best_epoch}")
                        
                        if stop:
                            break

                        if self.overfitter.check_stop(train_loss):
                            break

            self.early_stopping.restore(self.model)
            if self.early_stopping.triggered and self.early_stopping.best_params is not None:
                self.logger.subsection(f"Restored best parameters from epoch {self.early_stopping.best_epoch + 1}")

            return None, None, None

        finally:
            if self.resource_monitor is not None:
                self.resource_monitor.stop()

