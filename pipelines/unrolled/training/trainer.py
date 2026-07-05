from __future__ import annotations

from pathlib import Path

import torch

from models.unrolled          import TomoOperator
from tools.data.gaussians     import GaussianCurve
from tools.data.io            import FileIO


class UnrolledTrainer:
    LOSS_KINDS = ("l1", "mse")

    def __init__(self, model, model_cfg, x_axis, entry_config, ppg, run_dir: Path, logger, norm_stats):
        self.entry_config = entry_config
        self.training     = entry_config.training
        self.logger       = logger
        self.norm_stats   = norm_stats
        self.ppg          = ppg
        self.run_dir      = Path(run_dir)

        if entry_config.curve_loss not in self.LOSS_KINDS:
            raise ValueError(f"Unknown curve_loss '{entry_config.curve_loss}'. Available: {self.LOSS_KINDS}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.x_axis = torch.as_tensor(x_axis, dtype=torch.float32, device=self.device)
        self.dx     = float(self.x_axis[1] - self.x_axis[0])

        self.optimizer = torch.optim.AdamW(model_cfg.get_param_groups(self.model), betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.training.scheduler_epochs or self.training.epochs, eta_min=self.training.eta_min)

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
            "Sample points"   : int(self.x_axis.shape[0]),
            "Parameters"      : sum(p.numel() for p in self.model.parameters()),
        })

    @torch.no_grad()
    def _synthesise_measurements(self, gt_params: torch.Tensor, kz_map: torch.Tensor):
        gt_phys = self.norm_stats.denormalize_output(gt_params.float())
        curves  = GaussianCurve.reconstruct(gt_phys, self.x_axis, self.ppg).float()

        power = curves.sum(dim=1) * self.dx
        mask  = power > self.entry_config.power_floor

        target       = curves / power.clamp(min=self.entry_config.power_floor).unsqueeze(1)
        measurements = TomoOperator.forward(target, kz_map, self.x_axis, self.dx)

        noise_std = self.entry_config.measurement_noise_std
        if noise_std > 0.0:
            noise        = torch.randn_like(measurements.real) + 1j * torch.randn_like(measurements.real)
            measurements = measurements + noise_std / (2.0 ** 0.5) * noise

        return measurements, target, mask

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

    def _run_epoch(self, loader, train: bool) -> dict:
        self.model.train(train)

        total_loss = 0.0
        total_peak = 0.0
        n_batches  = 0

        for batch in loader:
            gt_params, kz_map = self._unpack(batch)
            measurements, target, mask = self._synthesise_measurements(gt_params, kz_map)

            with torch.set_grad_enabled(train):
                pred = self.model(measurements, kz_map, self.x_axis)
                loss = self._curve_loss(pred, target, mask)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.max_grad_norm)
                self.optimizer.step()

            total_loss += float(loss.detach())
            total_peak += float(self._peak_error_m(pred.detach(), target, mask))
            n_batches  += 1

        denominator = max(n_batches, 1)
        return {"loss": total_loss / denominator, "peak_mae_m": total_peak / denominator}

    def _save_checkpoint(self, name: str) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_dir / f"{name}.pt")

    def train(self, train_loader, val_loader, test_loader) -> dict:
        epochs_without_improvement = 0
        history : list[dict] = []

        for epoch in range(1, self.training.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, train=False)

            self.scheduler.step()

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            self.logger.info(f"epoch {epoch:>4} | train loss {train_metrics['loss']:.6f} | val loss {val_metrics['loss']:.6f} | val peak MAE {val_metrics['peak_mae_m']:.3f} m")

            if val_metrics["loss"] < self.best_val_loss - self.training.early_stop_min_delta:
                self.best_val_loss         = val_metrics["loss"]
                epochs_without_improvement = 0
                self._save_checkpoint("best")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > self.training.early_stop_patience:
                self.logger.subsection(f"Early stopping at epoch {epoch}: no val improvement for {epochs_without_improvement} epochs")
                break

        self._save_checkpoint("last")

        best_path = self.checkpoint_dir / "best.pt"
        if best_path.is_file():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        with torch.no_grad():
            self.test_metrics = self._run_epoch(test_loader, train=False)

        self.logger.section("[Test Metrics]")
        self.logger.kv_table(self.test_metrics)

        FileIO.save_json({"history": history, "test": self.test_metrics, "best_val_loss": self.best_val_loss}, self.run_dir / "training_summary.json")

        return {"history": history, "test": self.test_metrics}
