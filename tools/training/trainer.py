from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tools.monitoring.tracker          import Tracker
from tools.monitoring.resource_monitor import ResourceMonitor

from tools.training.aggregation import MetricAggregator
from tools.training.scheduling  import Scheduler, Warmup
from tools.training.stopping    import EarlyStopping, OverfitManager
from tools.training.gradients   import GradientClipper
from tools.training.checkpoint  import Checkpoint


class BaseTrainer:
    stage_name    = "Training"
    section_title = "[Training Loop]"

    def __init__(self, model, config, run_dir, logger, x_axis):
        self.logger = logger
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log_init_banner()

        self.run_dir         = Path(run_dir)
        self.checkpoint_path = self.run_dir / "best_model.pt"
        self.tracker         = Tracker(writer=self.config.io.writer, debug=config.training.log_debug)

        self.model  = model.to(self.device)
        self.x_axis = torch.tensor(x_axis, device=self.device, dtype=torch.float32)

        self.epochs                  = config.training.epochs
        self.validation_frequency    = config.training.validation_frequency
        self.accumulation_steps      = config.training.gradient_accumulation_steps
        self.use_amp                 = config.training.use_amp
        self.abort_on_nonfinite_loss = config.training.abort_on_nonfinite_loss

        if self.use_amp and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("use_amp=True autocasts to bfloat16, which this GPU does not support; disable use_amp on this device")

        param_groups = self._build_param_groups()

        lr_scale = self.config.optimizer.lr_scale
        if lr_scale != 1.0:
            for pg in param_groups:
                pg["lr"] = float(pg["lr"]) * lr_scale
            self.logger.subsection(f"Linear LR scaling x{lr_scale:.4f} applied to {len(param_groups)} param groups (batch-size rule).")

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

        self.criterion = self._build_criterion()

        self.global_step  = 0
        self.train_losses = []
        self.val_losses   = []

        self._update_optimizer(self.lr_scheduler.effective_lrs())

    def _log_init_banner(self) -> None:
        pass

    def _build_param_groups(self) -> list[dict]:
        raise NotImplementedError

    def _build_criterion(self):
        raise NotImplementedError

    def _compute_loss(self, batch) -> dict:
        raise NotImplementedError

    def _eval_step(self, batch, aggregator: MetricAggregator) -> dict:
        return self._compute_loss(batch)

    def _set_train_mode(self) -> None:
        self.model.train()

    def _on_optimizer_step(self) -> None:
        pass

    def _log_train_banner(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
        pass

    def _before_training(self, train_loader: DataLoader) -> None:
        pass

    def _before_epoch(self, epoch: int) -> None:
        pass

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        pass

    def _log_train_epoch_extra(self, avg_loss: float, epoch: int) -> None:
        pass

    def _build_optimizer(self, param_groups: list[dict]):
        betas        = tuple(self.config.optimizer.betas)
        eps          = self.config.optimizer.eps
        weight_decay = self.config.optimizer.weight_decay

        for pg in param_groups:
            pg.setdefault("betas",        betas)
            pg.setdefault("eps",          eps)
            pg.setdefault("weight_decay", weight_decay)

        optimizer = optim.AdamW(param_groups)
        return optimizer

    def _update_optimizer(self, lrs: list[float]):
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            param_group['lr'] = lr
            name              = param_group.get('name', str(i))
            self.tracker.log_scalar(f"lr/{name}", lr, self.global_step)

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def capture_state(self, epoch: int) -> dict:
        return {
            "epoch"  : epoch,
            "params" : self.model.state_dict(),
            "x_axis" : self.x_axis.cpu().numpy(),
        }

    def train_epoch(self, loader: DataLoader, epoch: int):
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum, n = 0.0, 0
        aggregator = MetricAggregator()
        n_batches  = len(loader)
        clear_n    = self.config.memory.clear_cache_every_n_steps

        window_has_grads = False

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]Epoch {epoch + 1}/{self.epochs}[/section] - train", total=n_batches)

            for batch_idx, batch in enumerate(loader):
                self._update_optimizer(self.lr_scheduler.effective_lrs())

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    loss_dict = self._compute_loss(batch)
                loss = loss_dict["total_loss"] / self.accumulation_steps

                if torch.isfinite(loss):
                    loss.backward()
                    window_has_grads = True

                    loss_sum += loss.item() * self.accumulation_steps
                    n += 1
                    aggregator.add(loss_dict)

                else:
                    if self.abort_on_nonfinite_loss:
                        raise FloatingPointError(f"{self.stage_name} loss is non-finite at step {self.global_step}")

                    self.logger.warning(f"{self.stage_name} loss is non-finite at step {self.global_step}; skipping batch (abort_on_nonfinite_loss disabled).")

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                    if window_has_grads:
                        grad_norm = self.grad_clipper.maybe_clip(self.model, self.global_step)
                        self.grad_clipper.record(grad_norm, self.global_step)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self._on_optimizer_step()
                        self.warmup.step()

                    window_has_grads = False

                self.global_step += 1
                self.tracker.set_step(self.global_step)

                if clear_n > 0 and self.global_step % clear_n == 0:
                    self._clear_cuda_cache()

                _prog.update(_task, advance=1)

        avg = loss_sum / max(1, n)
        self.tracker.log_scalar("loss/train", avg, epoch)
        self.tracker.log_metrics("loss_components/train", aggregator.reduce_components(), epoch)
        self.tracker.log_metrics("loss_weighted/train", aggregator.reduce_weighted(), epoch)

        monitor = aggregator.reduce_monitor()
        if monitor:
            self.tracker.log_metrics("loss_all/train", monitor, epoch)

        occupancy = aggregator.reduce_occupancy()
        if occupancy:
            self.tracker.log_metrics("occupancy/train", occupancy, epoch)

        self.tracker.log_memory(epoch)
        self._log_train_epoch_extra(avg, epoch)
        return avg

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage="validation"):
        self.model.eval()

        loss_sum, n = 0.0, 0
        aggregator  = MetricAggregator()

        try:
            with self.logger.track(transient=True) as _prog:
                _task = _prog.add_task(f"[section]Eval {stage}[/section] - epoch {epoch + 1}/{self.epochs}", total=len(loader))

                for batch in loader:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                        loss_dict = self._eval_step(batch, aggregator)
                    loss_sum += loss_dict["total_loss"].item()
                    n += 1
                    aggregator.add(loss_dict)

                    _prog.advance(_task)

            avg = loss_sum / max(1, n)
            self.tracker.log_metrics(f"loss_components/{stage}", aggregator.reduce_components(), epoch)
            self.tracker.log_metrics(f"loss_weighted/{stage}", aggregator.reduce_weighted(), epoch)
            if aggregator.monitor_sum:
                self.tracker.log_metrics(f"loss_all/{stage}", aggregator.reduce_monitor(), epoch)
            if aggregator.occupancy_sum:
                self.tracker.log_metrics(f"occupancy/{stage}", aggregator.reduce_occupancy(), epoch)
            if aggregator.extra_sum:
                self.tracker.log_metrics(f"permutation/{stage}", aggregator.reduce_extra(), epoch)
        finally:
            if self.config.memory.clear_cache_after_eval:
                self._clear_cuda_cache()

        return {"avg_loss": avg, "num_batches": n}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.logger.section(self.section_title)
        self._log_train_banner(train_loader, val_loader, test_loader)

        self._clear_cuda_cache()

        if self.resource_monitor is not None:
            self.resource_monitor.start()
        try:
            data_loader, val_loader, test_loader = self.overfitter.setup_loaders(train_loader, val_loader, test_loader)

            if self.overfitter.enabled:
                self.lr_scheduler.set_total_epochs(self.overfitter.planned_epochs())

            self._before_training(data_loader)

            with self.logger.live_monitor("Training Progress") as live_mon:
                with self.logger.track(transient=False) as _prog_epochs:
                    _task_epochs = _prog_epochs.add_task("[section]Training[/section]", total=self.epochs)

                    for epoch in range(self.epochs):
                        epoch_num = epoch + 1

                        self.logger.section(f"[Epoch {epoch_num}/{self.epochs}]")
                        self._before_epoch(epoch)

                        train_loss = self.train_epoch(data_loader, epoch)
                        self.logger.subsection(f"Train  : loss={train_loss:.4f}")

                        do_eval = (epoch_num % self.validation_frequency == 0) or (epoch_num == self.epochs)
                        if do_eval:
                            val      = self.evaluate(val_loader, epoch, stage="validation")
                            val_loss = val["avg_loss"]
                            self.logger.subsection(f"Validation : loss={val_loss:.4f}  (batches={val['num_batches']})")
                            self.tracker.log_scalar("loss/val", val_loss, epoch)
                            self.checkpoint.step(val_loss, epoch_num, self)
                            self.lr_scheduler.step(epoch)
                            stop = self.early_stopping(val_loss, epoch)
                            self._after_eval(val_loss, epoch)
                        else:
                            val_loss = float("nan")
                            self.lr_scheduler.step(epoch)
                            stop = False

                        self.train_losses.append(train_loss)
                        self.val_losses.append(val_loss)

                        if self.config.memory.clear_cache_after_epoch:
                            self._clear_cuda_cache()

                        effective_lrs = self.lr_scheduler.effective_lrs()
                        self._update_optimizer(effective_lrs)

                        live_mon.update(
                            epoch         = f"{epoch_num}/{self.epochs}",
                            train_loss    = train_loss,
                            val_loss      = val_loss,
                            best_val_loss = self.checkpoint.best_val_loss,
                            best_epoch    = self.checkpoint.best_epoch,
                            lr            = effective_lrs[0],
                        )
                        _prog_epochs.update(_task_epochs, advance=1, description=f"[section]Training[/section]  best_val={self.checkpoint.best_val_loss:.4f} @ ep {self.checkpoint.best_epoch}")

                        if stop or self.overfitter.check_stop(train_loss):
                            break

            if self.restore_best:
                self.checkpoint.restore_best(self.model, self.device)
            return self.train_losses, self.val_losses, self.checkpoint.best_val_loss
        finally:
            if self.resource_monitor is not None:
                self.resource_monitor.stop()
