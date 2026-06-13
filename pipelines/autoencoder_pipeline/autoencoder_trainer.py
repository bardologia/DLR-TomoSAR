from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tools                                 import Tracker, ResourceMonitor
from pipelines.training_pipeline.callbacks import Warmup, Scheduler, EarlyStopping, GradientClipper
from pipelines.training_pipeline.control   import Checkpoint, OverfitManager
from pipelines.training_pipeline.trainer   import MetricAggregator
from pipelines.autoencoder_pipeline.losses import ProfileAeLoss


class ProfileAeTrainer:
    def __init__(self, model, model_cfg, x_axis, config, run_dir, logger):
        self.logger = logger
        self.config = config

        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir         = Path(run_dir)
        self.checkpoint_path = self.run_dir / "best_model.pt"

        self.tracker = Tracker(writer=self.config.io.writer, debug=config.training.log_debug)

        self.model  = model.to(self.device)
        self.x_axis = torch.tensor(x_axis, device=self.device, dtype=torch.float32)

        self.epochs               = config.training.epochs
        self.validation_frequency = config.training.validation_frequency
        self.accumulation_steps   = config.training.gradient_accumulation_steps
        self.use_amp              = config.training.use_amp

        param_groups   = model_cfg.get_param_groups(self.model)
        self.base_lrs  = [float(pg["lr"]) for pg in param_groups]
        self.optimizer = self._build_optimizer(param_groups)

        self.warmup           = Warmup(config, self.logger, self.tracker)
        self.lr_scheduler     = Scheduler(self.base_lrs, self.warmup, config, self.logger, self.tracker)
        self.early_stopping   = EarlyStopping(config, self.logger, self.tracker)
        self.restore_best     = config.early_stopping.restore_best
        self.grad_clipper     = GradientClipper(config=config, logger=self.logger, tracker=self.tracker)
        self.checkpoint       = Checkpoint(self.logger, self.tracker, str(self.checkpoint_path))
        self.overfitter       = OverfitManager(config, self.logger)
        self.resource_monitor = ResourceMonitor(config=config.resources, logger=self.logger, tracker=self.tracker, step_getter=lambda: self.global_step)

        self.criterion = ProfileAeLoss(config.ae_loss)

        self.global_step  = 0
        self.train_losses = []
        self.val_losses   = []

        self._update_optimizer(self.lr_scheduler.effective_lrs())

    def _build_optimizer(self, param_groups):
        betas, eps, wd = tuple(self.config.optimizer.betas), self.config.optimizer.eps, self.config.optimizer.weight_decay
        for pg in param_groups:
            pg.setdefault("betas", betas)
            pg.setdefault("eps", eps)
            pg.setdefault("weight_decay", wd)
        return optim.AdamW(param_groups)

    def _update_optimizer(self, lrs):
        for i, (pg, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            pg["lr"] = lr
            self.tracker.log_scalar(f"lr/{pg.get('name', str(i))}", lr, self.global_step)

    def capture_state(self, epoch: int) -> dict:
        return {"epoch": epoch, "params": self.model.state_dict(), "x_axis": self.x_axis.cpu().numpy()}

    def _forward(self, curve):
        curve = curve.to(self.device).unsqueeze(-1).unsqueeze(-1)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            curve_hat, _ = self.model.reconstruct(curve)
            target       = self.model.normalize_curve(curve)
            loss_dict    = self.criterion(curve_hat, target)
        return loss_dict

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()
        n_batches   = len(loader)

        for batch_idx, curve in enumerate(loader):
            loss_dict = self._forward(curve)
            loss = loss_dict["total_loss"] / self.accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage-A loss is non-finite at step {self.global_step}")
            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                grad_norm = self.grad_clipper.maybe_clip(self.model, self.global_step)
                self.grad_clipper.record(grad_norm, self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self._update_optimizer(self.lr_scheduler.effective_lrs())
            self.warmup.step()
            self.global_step += 1
            self.tracker.set_step(self.global_step)

            loss_sum += loss.item() * self.accumulation_steps
            n += 1
            aggregator.add(loss_dict)

        avg = loss_sum / max(1, n)
        self.tracker.log_scalar("loss/train", avg, epoch)
        self.tracker.log_metrics("loss_components/train", aggregator.reduce_components(), epoch)
        self.tracker.log_metrics("loss_weighted/train", aggregator.reduce_weighted(), epoch)
        return avg

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage="validation"):
        self.model.eval()
        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()
        for curve in loader:
            loss_dict = self._forward(curve)
            loss_sum += loss_dict["total_loss"].item()
            n += 1
            aggregator.add(loss_dict)
        avg = loss_sum / max(1, n)
        self.tracker.log_metrics(f"loss_components/{stage}", aggregator.reduce_components(), epoch)
        self.tracker.log_metrics(f"loss_weighted/{stage}", aggregator.reduce_weighted(), epoch)
        return {"avg_loss": avg, "num_batches": n}

    def train(self, train_loader, val_loader, test_loader):
        self.logger.section("[Stage-A Autoencoder Training]")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.resource_monitor is not None:
            self.resource_monitor.start()
        try:
            data_loader, val_loader, test_loader = self.overfitter.setup_loaders(train_loader, val_loader, test_loader)

            for epoch in range(self.epochs):
                self.logger.section(f"[Epoch {epoch + 1}/{self.epochs}]")
                train_loss = self.train_epoch(data_loader, epoch)
                self.logger.subsection(f"Train : loss={train_loss:.4f}")

                do_eval = ((epoch + 1) % self.validation_frequency == 0) or (epoch + 1 == self.epochs)
                if do_eval:
                    val = self.evaluate(val_loader, epoch)
                    val_loss = val["avg_loss"]
                    self.logger.subsection(f"Validation : loss={val_loss:.4f}")
                    self.tracker.log_scalar("loss/val", val_loss, epoch)
                    self.checkpoint.step(val_loss, epoch + 1, self)
                    self.lr_scheduler.step(epoch)
                    stop = self.early_stopping(val_loss, epoch)
                else:
                    val_loss = float("nan")
                    self.lr_scheduler.step(epoch)
                    stop = False

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self._update_optimizer(self.lr_scheduler.effective_lrs())

                if stop or self.overfitter.check_stop(train_loss):
                    break

            if self.restore_best:
                self.checkpoint.restore_best(self.model, self.device)
            return self.train_losses, self.val_losses, self.checkpoint.best_val_loss
        finally:
            if self.resource_monitor is not None:
                self.resource_monitor.stop()
