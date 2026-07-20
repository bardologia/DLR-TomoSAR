from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import torch

from configuration.training           import SchedulerConfig, WarmupConfig
from pipelines.unrolled.synthesis     import MeasurementSynthesiser
from tools.data.io                    import FileIO
from tools.runtime.completion         import CompletionMarker
from tools.training                   import WeightEma
from tools.training.scheduling        import Scheduler, Warmup
from tools.training.vram_reservation  import VramReservation


@dataclass
class ScheduleSettings:
    warmup    : WarmupConfig
    scheduler : SchedulerConfig


class UnrolledTrainer:
    LOSS_KINDS = ("l1", "mse")

    def __init__(self, model, model_cfg, x_axis, entry_config, ppg, run_dir: Path, logger, norm_stats):
        self.entry_config = entry_config
        self.training     = entry_config.training
        self.logger       = logger
        self.norm_stats   = norm_stats
        self.run_dir      = Path(run_dir)

        if entry_config.curve_loss not in self.LOSS_KINDS:
            raise ValueError(f"Unknown curve_loss '{entry_config.curve_loss}'. Available: {self.LOSS_KINDS}")

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = model.to(self.device)
        self.x_axis  = torch.as_tensor(x_axis, dtype=torch.float32, device=self.device)
        self.use_amp = False

        self.synthesiser = MeasurementSynthesiser(self.x_axis, ppg, entry_config.power_floor, entry_config.measurement_noise_std)

        self.optimizer = torch.optim.AdamW(model_cfg.get_param_groups(self.model), betas=(0.9, 0.999), eps=1e-8)

        lr_scale = self.training.batch_size / self.training.lr_reference_batch_size if self.training.scale_lr_with_batch else 1.0
        if lr_scale != 1.0:
            for group in self.optimizer.param_groups:
                group["lr"] = float(group["lr"]) * lr_scale
            self.logger.subsection(f"Linear LR scaling x{lr_scale:.4f} applied to {len(self.optimizer.param_groups)} param groups (batch-size rule).")

        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

        schedules = ScheduleSettings(
            warmup    = WarmupConfig(warmup_steps=self.training.warmup_steps, warmup_start_factor=0.1, warmup_enabled=self.training.warmup_enabled, warmup_mode="linear"),
            scheduler = SchedulerConfig(type="cosine_annealing", epochs=self.training.scheduler_epochs or self.training.epochs, eta_min=self.training.eta_min),
        )

        self.warmup           = Warmup(schedules, logger)
        self.lr_scheduler     = Scheduler(self.base_lrs, self.warmup, schedules, logger)
        self.ema              = WeightEma(self.model, self.training.ema_decay, self.training.use_ema)
        self.vram_reservation = VramReservation(enabled=self.training.reserve_vram, keep_free_gb=self.training.vram_keep_free_gb, device=self.device, logger=logger)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.test_metrics : dict = {}

        self.logger.section("[Unrolled Trainer]")
        self.logger.kv_table({
            "Device"          : str(self.device),
            "Epochs"          : self.training.epochs,
            "Curve loss"      : entry_config.curve_loss,
            "Noise std"       : entry_config.measurement_noise_std,
            "Power floor"     : entry_config.power_floor,
            "Warmup"          : f"{self.training.warmup_enabled} ({self.training.warmup_steps} steps)",
            "EMA"             : f"{self.training.use_ema} (decay {self.training.ema_decay})",
            "VRAM reservation": f"{self.training.reserve_vram} (keep free {self.training.vram_keep_free_gb} GB)",
            "Sample points"   : int(self.x_axis.shape[0]),
            "Parameters"      : sum(p.numel() for p in self.model.parameters()),
        })

    @torch.no_grad()
    def _synthesise_measurements(self, gt_params: torch.Tensor, kz_map: torch.Tensor):
        gt_phys = self.norm_stats.denormalize_output(gt_params.float())

        return self.synthesiser.synthesise(gt_phys, kz_map)

    def _curve_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        per_pixel = diff.abs().mean(dim=1) if self.entry_config.curve_loss == "l1" else diff.pow(2).mean(dim=1)

        weights = mask.to(per_pixel.dtype)
        return (per_pixel * weights).sum() / weights.sum().clamp(min=1.0)

    @torch.no_grad()
    def _peak_error_m(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_peak   = self.x_axis[pred.argmax(dim=1)]
        target_peak = self.x_axis[target.argmax(dim=1)]

        weights = mask.to(pred_peak.dtype)
        return ((pred_peak - target_peak).abs() * weights).sum() / weights.sum().clamp(min=1.0)

    def _unpack(self, batch):
        if len(batch) < 3 or batch[2] is None:
            raise RuntimeError("Unrolled training requires the per-pixel kz geometry field; the dataset pipeline must run with build_geometry_field=True")

        gt_params = batch[1].to(self.device, non_blocking=True)
        kz_map    = batch[2].to(self.device, non_blocking=True).float()

        return gt_params, kz_map

    def _compute_loss(self, batch) -> dict:
        gt_params, kz_map = self._unpack(batch)
        measurements, target, mask = self._synthesise_measurements(gt_params, kz_map)

        pred = self.model(measurements, kz_map, self.x_axis)
        loss = self._curve_loss(pred, target, mask)

        return {"total_loss": loss, "pred": pred, "target": target, "mask": mask}

    def _apply_lrs(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def _run_epoch(self, loader, train: bool) -> dict:
        self.model.train(train)

        total_loss = 0.0
        total_peak = 0.0
        n_batches  = 0

        for batch in loader:
            with torch.set_grad_enabled(train):
                losses = self._compute_loss(batch)
                loss   = losses["total_loss"]

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.max_grad_norm)
                self.optimizer.step()
                self.ema.update(self.model)
                self.warmup.step()
                self._apply_lrs(self.lr_scheduler.effective_lrs())

            total_loss += float(loss.detach())
            total_peak += float(self._peak_error_m(losses["pred"].detach(), losses["target"], losses["mask"]))
            n_batches  += 1

        denominator = max(n_batches, 1)
        return {"loss": total_loss / denominator, "peak_mae_m": total_peak / denominator}

    def _save_checkpoint(self, name: str) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_dir / f"{name}.pt")

    def train(self, train_loader, val_loader, test_loader) -> dict:
        CompletionMarker.clear(self.run_dir)
        self.vram_reservation.fill()

        epochs_without_improvement = 0
        history : list[dict] = []

        for epoch in range(1, self.training.epochs + 1):
            self.lr_scheduler.step(epoch - 1)
            self._apply_lrs(self.lr_scheduler.effective_lrs())

            train_metrics = self._run_epoch(train_loader, train=True)
            with torch.no_grad(), self.ema.applied(self.model):
                val_metrics = self._run_epoch(val_loader, train=False)

                improved = val_metrics["loss"] < self.best_val_loss - self.training.early_stop_min_delta
                if improved:
                    self.best_val_loss         = val_metrics["loss"]
                    epochs_without_improvement = 0
                    self._save_checkpoint("best")

            self.vram_reservation.refill()

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            self.logger.info(f"epoch {epoch:>4} | train loss {train_metrics['loss']:.6f} | val loss {val_metrics['loss']:.6f} | val peak MAE {val_metrics['peak_mae_m']:.3f} m")

            if not improved:
                epochs_without_improvement += 1

            if epochs_without_improvement > self.training.early_stop_patience:
                self.logger.subsection(f"Early stopping at epoch {epoch}: no val improvement for {epochs_without_improvement} epochs")
                break

        with self.ema.applied(self.model):
            self._save_checkpoint("last")

        best_path = self.checkpoint_dir / "best.pt"
        if best_path.is_file():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        with torch.no_grad():
            self.test_metrics = self._run_epoch(test_loader, train=False)

        self.logger.section("[Test Metrics]")
        self.logger.kv_table(self.test_metrics)

        FileIO.save_json({"history": history, "test": self.test_metrics, "best_val_loss": self.best_val_loss}, self.run_dir / "training_summary.json")

        CompletionMarker.stamp(self.run_dir, {
            "stage"            : "training",
            "epochs_completed" : len(history),
            "epochs_total"     : self.training.epochs,
            "best_val_loss"    : float(self.best_val_loss),
        })

        return {"history": history, "test": self.test_metrics}
