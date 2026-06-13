from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tools                                 import Tracker, ResourceMonitor, PermutationMetrics
from pipelines.training_pipeline.callbacks import Warmup, Scheduler, EarlyStopping, GradientClipper
from pipelines.training_pipeline.control   import Checkpoint, OverfitManager, CurriculumController
from pipelines.training_pipeline.loss      import Loss
from pipelines.training_pipeline.trainer   import MetricAggregator
from pipelines.jepa_pipeline.coupling      import StageAMode, TargetProvider
from pipelines.jepa_pipeline.losses        import JepaLoss


class JepaModule(nn.Module):
    def __init__(self, backbone: nn.Module, autoencoder: nn.Module) -> None:
        super().__init__()
        self.backbone    = backbone
        self.autoencoder = autoencoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class JepaPredictorTrainer:
    def __init__(self, model: JepaModule, backbone_cfg, x_axis, config, run_dir, logger, norm_stats):
        self.logger       = logger
        self.config       = config
        self.backbone_cfg = backbone_cfg
        self.gaussian_cfg = config.gaussian
        self.curriculum   = config.curriculum
        self.norm_stats   = norm_stats

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(run_dir)
        self.checkpoint_path = self.run_dir / "best_model.pt"

        self.tracker = Tracker(writer=self.config.io.writer, debug=config.training.log_debug)

        self.model  = model.to(self.device)
        self.x_axis = torch.tensor(x_axis, device=self.device, dtype=torch.float32)

        self.stage_a_mode = StageAMode(config.stage_a_mode)
        self.stage_a_mode.apply(self.model.autoencoder)

        self.epochs               = config.training.epochs
        self.validation_frequency = config.training.validation_frequency
        self.accumulation_steps   = config.training.gradient_accumulation_steps
        self.use_amp              = config.training.use_amp

        param_groups = backbone_cfg.get_param_groups(self.model.backbone)
        param_groups += self.stage_a_mode.param_groups(self.model.autoencoder, config.ae_finetune_lr, config.ae_finetune_wd)
        self.base_lrs  = [float(pg["lr"]) for pg in param_groups]
        self.optimizer = self._build_optimizer(param_groups)

        self.warmup         = Warmup(config, self.logger, self.tracker)
        self.lr_scheduler   = Scheduler(self.base_lrs, self.warmup, config, self.logger, self.tracker)
        self.early_stopping = EarlyStopping(config, self.logger, self.tracker)
        self.restore_best   = config.early_stopping.restore_best
        self.grad_clipper   = GradientClipper(config=config, logger=self.logger, tracker=self.tracker)
        self.checkpoint     = Checkpoint(self.logger, self.tracker, str(self.checkpoint_path))
        self.overfitter     = OverfitManager(config, self.logger)
        self.resource_monitor   = ResourceMonitor(config=config.resources, logger=self.logger, tracker=self.tracker, step_getter=lambda: self.global_step)
        self.permutation_metrics = PermutationMetrics(config.permutation_metrics, logger=self.logger)

        target_provider = TargetProvider(config.target_provider, self.model.autoencoder.encoder, config.ema_decay).to(self.device)
        inner = Loss(self.x_axis, self.logger, self.tracker, self.gaussian_cfg, self.curriculum.warmup,
                     norm_stats=self.norm_stats, geometry_cfg=config.geometry, log_all_losses=config.training.log_all_losses)
        self.criterion = JepaLoss(self.model.autoencoder, inner, target_provider, config.embedding_loss, self.norm_stats)

        self.curriculum_controller = CurriculumController(
            curriculum=self.curriculum, criterion=self.criterion, early_stopping=self.early_stopping,
            lr_scheduler=self.lr_scheduler, warmup=self.warmup, optimizer=self.optimizer,
            update_optimizer=self._update_optimizer, logger=self.logger,
        )

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
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
            "x_axis" : self.x_axis.cpu().numpy(),
        }

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.model.backbone.train()
        if self.stage_a_mode.trainable:
            self.model.autoencoder.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()
        n_batches   = len(loader)

        for batch_idx, batch in enumerate(loader):
            images    = batch[0].to(self.device)
            gt_params = batch[1].to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                z_hat     = self.model(images)
                loss_dict = self.criterion(z_hat, gt_params)
                loss      = loss_dict["total_loss"] / self.accumulation_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage-B loss is non-finite at step {self.global_step}")
            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                grad_norm = self.grad_clipper.maybe_clip(self.model, self.global_step)
                self.grad_clipper.record(grad_norm, self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.stage_a_mode.trainable:
                    self.criterion.target_provider.update(self.model.autoencoder.encoder)

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
        for batch in loader:
            images    = batch[0].to(self.device)
            gt_params = batch[1].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                z_hat     = self.model(images)
                loss_dict = self.criterion(z_hat, gt_params)
            loss_sum += loss_dict["total_loss"].item()
            n += 1
            aggregator.add(loss_dict)

            pred_params = self.criterion.decode_params(z_hat)
            pred_phys   = self.norm_stats.denormalize_output(pred_params.float())
            gt_phys     = self.norm_stats.denormalize_output(gt_params.float())
            perm_m      = self.permutation_metrics.compute(pred_phys, gt_phys, self.gaussian_cfg.params_per_gaussian)
            aggregator.add_extra(perm_m)

        avg = loss_sum / max(1, n)
        self.tracker.log_metrics(f"loss_components/{stage}", aggregator.reduce_components(), epoch)
        self.tracker.log_metrics(f"loss_weighted/{stage}", aggregator.reduce_weighted(), epoch)
        if aggregator.extra_sum:
            self.tracker.log_metrics(f"permutation/{stage}", aggregator.reduce_extra(), epoch)
        return {"avg_loss": avg, "num_batches": n}

    def train(self, train_loader, val_loader, test_loader):
        self.logger.section("[Stage-B JEPA Predictor Training]")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.resource_monitor is not None:
            self.resource_monitor.start()
        try:
            data_loader, val_loader, test_loader = self.overfitter.setup_loaders(train_loader, val_loader, test_loader)

            for epoch in range(self.epochs):
                self.logger.section(f"[Epoch {epoch + 1}/{self.epochs}]")
                self.curriculum_controller.maybe_swap(epoch)
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
