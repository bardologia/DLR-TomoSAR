from __future__ import annotations

import os
os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]     = "4"

import gc
import torch
import torch.nn.functional as F
from pathlib import Path

from tools import ModelSummary, ShapeLogger, Tracker, ResourceMonitor

from pipelines.training_pipeline.early_stopping import EarlyStopping
from pipelines.training_pipeline.warmup import Warmup
from pipelines.training_pipeline.scheduler import Scheduler
from pipelines.training_pipeline.ema import EMA
from pipelines.training_pipeline.gradient_clipper import GradientClipper
from pipelines.training_pipeline.overfit_manager import OverfitManager
from pipelines.training_pipeline.checkpoint import Checkpoint
from pipelines.training_pipeline.loss import Loss
from pipelines.training_pipeline.metrics import Metrics
from pipelines.training_pipeline.metrics_aggregator import MetricsAggregator


class Trainer:
    def __init__(self, model, x_axis, config, run_dir, logger, norm_stats=None):
        self.logger       = logger
        self.config       = config
        self.gaussian_cfg = config.gaussian
        self.loss_cfg     = config.loss
        self.norm_stats   = norm_stats
        
        self.logger.section("[Training Start]")
        self.logger.subsection(f"Device Name   : {torch.cuda.get_device_name(0)}")
        self.logger.subsection(f"Total Memory  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.subsection(f"CUDA Version  : {torch.version.cuda}")
        self.logger.subsection(f"Log Directory : {self.config.io.logdir} \n")
        
        self.checkpoint_path = Path(run_dir) / "best_model.pt"
        self.tracker         = Tracker(writer=self.config.io.writer, debug=getattr(self.config.training, "log_debug", False))
        
        self.device = self.config.training.device
        self.model  = model.to(self.device)
        self.x_axis = x_axis.to(self.device)

        self.epochs               = self.config.training.epochs
        self.validation_frequency = self.config.training.validation_frequency

        self.use_amp = self.config.training.use_amp and torch.cuda.is_available()
        self.scaler  = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.accumulation_steps = self.config.training.gradient_accumulation_steps
        self.param_groups       = self.make_param_groups()
        self.optimizer          = torch.optim.AdamW(self.param_groups, betas = self.config.optimizer.betas, eps = self.config.optimizer.eps)

        self.warmup         = Warmup(self.optimizer, self.config, self.logger, self.tracker)
        self.ema            = EMA(self.model, self.config, self.logger, self.tracker)
        self.early_stopping = EarlyStopping(self.config, self.logger, self.tracker)
        self.lr_scheduler   = Scheduler(self.optimizer, self.warmup, self.config, self.logger, self.tracker)
        self.metrics        = Metrics(self.config.training.verbose, self.tracker, logger=self.logger, x_axis=self.x_axis, gaussian_cfg=self.gaussian_cfg)
        self.criterion      = Loss(self.x_axis, self.logger, self.tracker, self.metrics.reconstruct_gaussians, self.gaussian_cfg, self.loss_cfg, norm_stats=self.norm_stats)
        self.checkpoint     = Checkpoint(self.logger, self.tracker, str(self.checkpoint_path))
        self.shape_logger   = ShapeLogger(model = self.model, logger = self.logger, include_types=self.model.config.shape_logger_types, docs_dir=self.config.io.docs_dir or self.config.io.logdir).attach()
        self.summary        = ModelSummary(logger = self.logger, model = self.model, docs_dir=self.config.io.docs_dir or self.config.io.logdir)
        self.summary.run()
        self.summary.save_markdown()
        
        self.global_step   = 0
        self.train_losses  = []
        self.val_losses    = []
        
        self.current_epoch = 0
        self.grad_clipper  = GradientClipper(
            initial_threshold = self.config.training.max_grad_norm,
            logger            = self.logger,
            tracker           = self.tracker,
        )

        self.overfitter = OverfitManager(self.config, self.logger)

        self.resource_monitor = ResourceMonitor(
            config     = self.config.resources,
            logger     = self.logger,
            tracker    = self.tracker,
            step_getter= lambda: self.global_step,
        )
              
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def make_param_groups(self):
        param_groups = self.model.config.get_param_groups(self.model)

        self.logger.section("[Optimizer Parameter Groups]")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            self.logger.subsection(f"{group['name']} - LR: {group['lr']}, Weight Decay: {group['weight_decay']}, Parameters: {num_params:,}")

        return param_groups
    
    def _apply_amplitude_constraint(self, params: torch.Tensor) -> torch.Tensor:
        ppg = self.gaussian_cfg.params_per_gaussian
        amplitude_indices = list(range(0, params.shape[1], ppg))
        out = params.clone()
        out[:, amplitude_indices] = F.softplus(params[:, amplitude_indices])
        return out

    def forward(self, images, exp_curves, step, gt_params=None):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_params = self.model(images)  # amplitude constraint applied inside model.forward()
            loss_dict   = self.criterion(pred_params, exp_curves, step, gt_params=gt_params)
            self.shape_logger.detach()

        return pred_params, loss_dict
    
    def backward(self, loss, step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if step:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            grad_norm = self.grad_clipper.step(self.model.parameters(), self.global_step)
            self.tracker.log_scalar("train/grad_norm", grad_norm, self.global_step)
            if self.grad_clipper.threshold is not None:
                self.tracker.log_scalar("train/grad_clip_thr", self.grad_clipper.threshold, self.global_step)

            if self.tracker.debug and self.global_step % 100 == 0:
                self.tracker.log_gradients(self.model, self.global_step, max_grad_norm=self.grad_clipper.threshold)
                self.tracker.log_optimizer(self.optimizer, self.global_step)
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
            self.warmup.step()
            self.global_step += 1
            self.ema.update(self.model, step=self.global_step)

    def train_epoch(self, train_loader, epoch):
        self.current_epoch = epoch
        self.model.train()
        total_loss   = 0
        num_batches  = 0
        n_samples    = 0
        batch_losses = []

        activation_hooks = []
        step_box = [self.global_step]
        if self.tracker.debug and epoch > 0 and epoch % 10 == 0:
            activation_hooks = self.tracker.log_activations(self.model, step_box)

        try:
            with self.logger.track(transient=True) as _prog:
                _task = _prog.add_task(f"[section]Epoch {epoch+1}/{self.epochs}[/section] - train", total=len(train_loader))
                for batch_idx, batch in enumerate(train_loader):
                    images, exp_curves, gt_params = self._unpack_batch(batch)
                    
                    images     = images.to(self.device, non_blocking=True)
                    exp_curves = exp_curves.to(self.device, non_blocking=True)
                    if gt_params is not None:
                        gt_params = gt_params.to(self.device, non_blocking=True)

                    step_box[0] = self.global_step
                    pred_params, loss_dict = self.forward(images, exp_curves, self.global_step, gt_params=gt_params)
                    loss = loss_dict["total_loss"] / self.accumulation_steps

                    should_step = (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) >= len(train_loader)
                    self.backward(loss, step=should_step)

                    loss_value   = loss.item() * self.accumulation_steps
                    total_loss  += loss_value
                    num_batches += 1
                    n_samples   += images.shape[0]
                    batch_losses.append(loss_value)

                    self.tracker.log_scalar("loss/train_step", loss_value, self.global_step)

                    clear_n = self.config.memory.clear_cache_every_n_steps
                    if clear_n > 0 and self.global_step > 0 and self.global_step % clear_n == 0:
                        del pred_params, loss, loss_dict
                        self._clear_memory()
                    
                    _prog.update(_task, advance=1, description=f"[section]Epoch {epoch+1}/{self.epochs}[/section] - train  loss={loss_value:.4f}")

                    if activation_hooks:
                        for hook in activation_hooks:
                            hook.remove()
                        activation_hooks = []
        finally:
            for hook in activation_hooks:
                hook.remove()

        avg_loss = total_loss / max(1, num_batches)

        self.tracker.log_scalar("loss/train_epoch", avg_loss, epoch)
        self.tracker.log_memory(epoch)

        if self.tracker.debug and epoch > 0 and epoch % 10 == 0:
            self.tracker.log_weights(self.model, epoch)
        
        return avg_loss
                
    def evaluate(self, loader, epoch, stage="validation", deep: bool | None = None):
        if deep is None:
            deep = self.config.training.deep_validation
        mem_cfg = self.config.memory

        self.model.eval()
        self.ema.apply_to(self.model)

        agg = MetricsAggregator(
            self.metrics,
            deep              = deep,
            gaussian_cfg      = self.gaussian_cfg,
            keep_pixel_arrays = mem_cfg.eval_keep_pixel_arrays,
            pixel_subsample   = mem_cfg.eval_pixel_subsample,
        )

        try:
            total_loss  = 0.0
            num_batches = 0
            last_C      = 0

            with torch.no_grad():
                with self.logger.track(transient=True) as _prog:
                    _task = _prog.add_task(f"[section]Eval {stage}[/section] - epoch {epoch+1}/{self.epochs}", total=len(loader))
                    for batch in loader:
                        images, exp_curves, gt_params = self._unpack_batch(batch)
                        
                        images     = images.to(self.device, non_blocking=True)
                        exp_curves = exp_curves.to(self.device, non_blocking=True)
                        if gt_params is not None:
                            gt_params = gt_params.to(self.device, non_blocking=True)

                        pred_params, loss_dict = self.forward(images, exp_curves, epoch)
                        total_loss += loss_dict["total_loss"].item()
                        last_C      = pred_params.shape[1]

                        gt_params_for_metrics = gt_params
                        if (
                            deep
                            and gt_params is not None
                            and self.norm_stats is not None
                            and self.norm_stats.stats.output_stats is not None
                        ):
                            gt_params_for_metrics = self.norm_stats.denormalize_output(gt_params)

                        agg.update(pred_params, exp_curves, gt_params=gt_params_for_metrics if deep else None)
                        num_batches += 1

                        del images, exp_curves, pred_params, loss_dict
                        del gt_params
                        
                        _prog.advance(_task)

            avg_loss = total_loss / max(1, num_batches)
            metrics  = agg.finalize(epoch, stage, last_C)
        
        finally:
            self.ema.restore(self.model)
            del agg
            if mem_cfg.clear_cache_after_eval:
                self._clear_memory()

        results = {
            "avg_loss"     : avg_loss,
            "metrics"      : metrics,
            "num_batches"  : num_batches,
            "deep"         : deep,
        }
        
        return results

    def train(self, train_loader, val_loader, test_loader):    
        self.logger.section("[Training Loop]")
        self.logger.subsection(f"Train loader size      = {len(train_loader)}")
        self.logger.subsection(f"Validation loader size = {len(val_loader)}")
        self.logger.subsection(f"Test loader size       = {len(test_loader)}")
        
        self._clear_memory()
        self.optimizer.zero_grad()

        if self.resource_monitor is not None:
            self.resource_monitor.start()

        try:
            data_loader, eval_train_loader, val_loader, test_loader = self.overfitter.setup_loaders(train_loader, val_loader, test_loader)
        
            epochs = self.epochs
            with self.logger.live_monitor("Training Progress") as live_mon:
                with self.logger.track(transient=False) as _prog_epochs:
                    _task_epochs = _prog_epochs.add_task("[section]Training[/section]", total=epochs)
                    for epoch in range(epochs):
                        epoch_num  = epoch + 1

                        self.logger.section(f"[Epoch {epoch_num}/{epochs}]")
                        train_loss = self.train_epoch(data_loader, epoch)
                        self.logger.subsection(f"Train  : loss={train_loss:.4f}")

                        val_results  = self.evaluate(val_loader, epoch, stage="validation")
                        val_loss     = val_results["avg_loss"]
                        
                        self.train_losses.append(val_loss)
                        self.val_losses.append(val_loss)
                    
                        self.logger.subsection(f"Validation : loss={val_loss:.4f}  (batches={val_results['num_batches']})")

                        train_results = self.evaluate(eval_train_loader, epoch, stage="train")
                        self.logger.subsection(f"Train eval : loss={train_results['avg_loss']:.4f}")
              
                        if self.config.memory.clear_cache_after_epoch:
                            self._clear_memory()

                        self.tracker.log_scalar("loss/val",        val_loss,                   epoch)
                        self.tracker.log_scalar("loss/train_eval", train_results["avg_loss"],  epoch)

                        self.checkpoint.step(val_loss, epoch_num, val_results["metrics"], self)
                        self.lr_scheduler.step(epoch, metric=val_loss)
                    
                        stop = self.early_stopping(val_loss, self.model, epoch)
                    
                        monitor_data = {
                            "epoch"         : f"{epoch_num}/{epochs}",
                            "train_loss"    : train_loss,
                            "val_loss"      : val_loss,
                            "best_val_loss" : self.checkpoint.best_val_loss,
                            "best_epoch"    : self.checkpoint.best_epoch,
                            "lr"            : self.optimizer.param_groups[0]["lr"],
                        }        
                        live_mon.update(**monitor_data)

                        _prog_epochs.update(_task_epochs, advance=1, description=f"[section]Training[/section]  best_val={self.checkpoint.best_val_loss:.4f} @ ep {self.checkpoint.best_epoch}",)

                        if stop:
                            break

                        if self.overfitter.check_stop(train_loss):
                            break

            self.shape_logger.save_markdown(sort_by_layer=True)
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            self.logger.section("[Final Evaluation]")
            self.logger.subsection(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")

            train_final_results      = self.evaluate(train_loader, checkpoint["epoch"], stage="final_train", deep=True)
            self.logger.subsection(f"Train      : loss={train_final_results['avg_loss']:.4f}")
            validation_final_results = self.evaluate(val_loader,   checkpoint["epoch"], stage="final_validation", deep=True)
            self.logger.subsection(f"Validation : loss={validation_final_results['avg_loss']:.4f}")
            test_final_results = self.evaluate(test_loader, checkpoint["epoch"], stage="final_test", deep=True)
            self.logger.subsection(f"Test       : loss={test_final_results['avg_loss']:.4f}")
        
            return train_final_results, validation_final_results, test_final_results

        finally:
            if self.resource_monitor is not None:
                self.resource_monitor.stop()
