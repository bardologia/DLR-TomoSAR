from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from tools                            import PermutationMetrics
from tools.training        import BaseTrainer, MetricAggregator
from pipelines.backbone.training.loss import Loss
from pipelines.backbone.training.docs import TrainingDocs


class CurriculumController:
    def __init__(self, curriculum, criterion, early_stopping, lr_scheduler, warmup, optimizer, update_optimizer, logger):
        self.curriculum       = curriculum
        self.criterion        = criterion
        self.early_stopping   = early_stopping
        self.lr_scheduler     = lr_scheduler
        self.warmup           = warmup
        self.optimizer        = optimizer
        self.update_optimizer = update_optimizer
        self.logger           = logger

    def maybe_swap(self, epoch: int) -> bool:
        lc = self.curriculum
        if not (lc.enabled and epoch == lc.swap_epoch):
            return False

        self.logger.section(f"[Curriculum Loss Swap @ epoch {epoch + 1}]")
        self.criterion.set_curriculum(lc.complete)
        self.logger.subsection("Loss config replaced with curriculum.loss.complete.")
        self.logger.subsection(f"Param match strategy updated to '{lc.complete.param_match}'.")

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


class Trainer(BaseTrainer):
    stage_name    = "Total"
    section_title = "[PyTorch Training Loop]"

    def __init__(self, model, model_cfg, x_axis, config, run_dir, logger, norm_stats=None, emit_docs=True):
        self.model_cfg       = model_cfg
        self.gaussian_cfg    = config.gaussian
        self.curriculum      = config.curriculum
        self.warmup_loss_cfg = config.curriculum.warmup
        self.norm_stats      = norm_stats
        self.emit_docs       = emit_docs

        super().__init__(model, config, run_dir, logger, x_axis)

        self.docs                = TrainingDocs(self.model, self.model_cfg, self.logger, self.run_dir, enabled=self.emit_docs)
        self.permutation_metrics = PermutationMetrics(self.config.permutation_metrics, logger=self.logger)

        self.curriculum_controller = CurriculumController(
            curriculum       = self.curriculum,
            criterion        = self.criterion,
            early_stopping   = self.early_stopping,
            lr_scheduler     = self.lr_scheduler,
            warmup           = self.warmup,
            optimizer        = self.optimizer,
            update_optimizer = self._update_optimizer,
            logger           = self.logger,
        )

    def _log_init_banner(self) -> None:
        self.logger.section("[Training Start]")
        self.logger.kv_table({
            "Backend":       "PyTorch",
            "Device":        self.device,
            "Log Directory": self.config.io.logdir,
        })

    def _build_param_groups(self) -> list[dict]:
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self):
        return Loss(self.x_axis, self.logger, self.tracker, self.gaussian_cfg, self.warmup_loss_cfg, norm_stats=self.norm_stats, geometry_cfg=self.config.geometry, log_all_losses=self.config.training.log_all_losses)

    def _warn_nonfinite(self, tensor: torch.Tensor, name: str) -> None:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.logger.warning(f"NaN or Inf detected in {name} at step {self.global_step}!")

    def _compute_loss(self, batch) -> dict:
        images    = batch[0].to(self.device)
        gt_params = batch[1].to(self.device) if len(batch) > 1 and batch[1] is not None else None

        if self.tracker.debug:
            self._warn_nonfinite(images, "input images")
            if gt_params is not None:
                self._warn_nonfinite(gt_params, "ground truth parameters")

        pred_params = self.model(images)
        if self.tracker.debug:
            self._warn_nonfinite(pred_params, "model predictions")

        return self.criterion(pred_params, gt_params)

    def _eval_step(self, batch, aggregator: MetricAggregator) -> dict:
        images    = batch[0].to(self.device)
        gt_params = batch[1].to(self.device) if len(batch) > 1 and batch[1] is not None else None

        pred_params = self.model(images)
        loss_dict   = self.criterion(pred_params, gt_params)

        if gt_params is not None:
            pred_phys = self.norm_stats.denormalize_output(pred_params)
            gt_phys   = self.norm_stats.denormalize_output(gt_params)

            perm_m = self.permutation_metrics.compute(pred_phys.float(), gt_phys.float(), self.gaussian_cfg.params_per_gaussian)
            aggregator.add_extra(perm_m)

        return loss_dict

    def _log_train_banner(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        self.logger.subsection(f"Train loader size      = {len(train_loader)}")
        self.logger.subsection(f"Validation loader size = {len(val_loader)}")
        self.logger.subsection(f"Test loader size       = {len(test_loader)}")
        self.logger.subsection(f"Device                 = {self.device} \n")

    def _before_training(self, train_loader: DataLoader) -> None:
        self.docs.emit(train_loader, self.device)

    def _before_epoch(self, epoch: int) -> None:
        swapped = self.curriculum_controller.maybe_swap(epoch)
        if swapped and not self.curriculum.reset_early_stopping:
            self.logger.warning("Curriculum loss swapped but early stopping not reset; best_val_loss spans two loss scales and is not comparable across the swap epoch.")

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        phase = self._curriculum_phase(epoch)
        self.tracker.log_scalar(f"loss_phase/{phase}/val", val_loss, epoch)
        self._trial_callback(val_loss, epoch)

    def _log_train_epoch_extra(self, avg_loss: float, epoch: int) -> None:
        phase = self._curriculum_phase(epoch)
        self.tracker.log_scalar(f"loss_phase/{phase}/train", avg_loss, epoch)

    def _curriculum_phase(self, epoch: int) -> str:
        if not self.curriculum.enabled:
            return "complete"

        return "warmup" if epoch < self.curriculum.swap_epoch else "complete"

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        pass

    def maybe_run_loss_probe(self, train_loader, probe_config=None) -> None:
        if probe_config is None or not probe_config.enabled:
            return

        from pipelines.backbone.training.loss_probe import LossScaleProbe

        probe = LossScaleProbe(
            probe_cfg    = probe_config,
            loss_cfg     = self.warmup_loss_cfg,
            gaussian_cfg = self.gaussian_cfg,
            geometry_cfg = self.config.geometry,
            norm_stats   = self.norm_stats,
            logger       = self.logger,
        )
        probe.run(train_loader, self.model, self.device, self.x_axis)
