from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tools.logger import Logger

from .config  import AutoencoderConfig, TrainerConfig
from .losses  import CompositeLoss
from .model   import Autoencoder
from .plotter import Plotter


class MetricMeter:

    def __init__(self) -> None:
        self._sums  : dict[str, float] = {}
        self._count : int = 0

    def update(self, losses: dict[str, torch.Tensor], batch_size: int) -> None:
        for k, v in losses.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v.item()) * batch_size
        self._count += batch_size

    def compute(self, prefix: str = "") -> dict[str, float]:
        if self._count == 0:
            return {}
        return {f"{prefix}{k}": v / self._count for k, v in self._sums.items()}


class Trainer:

    def __init__(
        self,
        model         : Autoencoder,
        ae_config     : AutoencoderConfig,
        train_loader  : DataLoader,
        val_loader    : DataLoader | None,
        run_directory : Path,
        logger        : Logger,
        plotter       : Plotter,
        writer        : SummaryWriter | None = None,
    ) -> None:
        self.cfg          = ae_config.trainer
        self.ae_config    = ae_config
        self.device       = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == "cpu" else "cpu")
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.run_dir      = Path(run_directory)
        self.logger       = logger
        self.writer       = writer
        self.plotter      = plotter

        self.criterion = CompositeLoss(ae_config.loss).to(self.device)
        self.optimizer = self._build_optimizer(model, self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.scaler    = torch.amp.GradScaler(self.device.type, enabled=self.cfg.use_amp and self.device.type == "cuda")

        self.checkpoint_dir = Path(ae_config.io.checkpoint_dir or self.run_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history           : list[dict[str, float]] = []
        self.best_val_total    : float                  = math.inf
        self.best_epoch        : int                    = 0
        self.epochs_no_improve : int                    = 0
        self.global_step       : int                    = 0

        self._csv_path = Path(ae_config.io.logs_dir or self.run_dir / "logs") / "training_metrics.csv"
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_header_written = False

    @staticmethod
    def _build_optimizer(model: Autoencoder, config: TrainerConfig) -> torch.optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found.")
        name = config.optimizer.lower()
        if name == "adam" : return torch.optim.Adam (params, lr=config.learning_rate, weight_decay=config.weight_decay)
        if name == "adamw": return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        if name == "sgd"  : return torch.optim.SGD  (params, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        raise ValueError(f"Unknown optimizer '{name}'.")

    @staticmethod
    def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainerConfig):
        name = config.scheduler.lower()
        if name == "none"  : return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, **config.scheduler_kwargs)
        if name == "step"  :
            return torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler_kwargs)
        raise ValueError(f"Unknown scheduler '{name}'.")

    def fit(self) -> list[dict[str, float]]:
        self.logger.section("[Autoencoder Training]")
        with self.logger.live_monitor("Autoencoder Training") as live_mon:
            with self.logger.track(transient=False) as _prog:
                _task = _prog.add_task("[section]AE Training[/section]", total=self.cfg.epochs)
                for epoch in range(1, self.cfg.epochs + 1):
                    train_metrics = self._train_one_epoch(epoch)
                    val_metrics   : dict[str, float] = {}
                    if self.val_loader is not None and (epoch % self.cfg.val_every == 0):
                        val_metrics = self._validate(epoch)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    record = {"epoch": epoch, "lr": self._current_lr(), **train_metrics, **val_metrics}
                    self.history.append(record)
                    self._log_record(record)
                    self._append_csv(record)

                    improved = self._update_best(val_metrics, train_metrics, epoch)
                    if epoch % self.cfg.save_every == 0:
                        self._save_checkpoint(f"epoch_{epoch}")

                    # Update live monitor
                    monitor_data = {
                        "epoch": f"{epoch}/{self.cfg.epochs}",
                        "train_loss": train_metrics.get("train/total", 0.0),
                        "best_val_total": self.best_val_total,
                        "best_epoch": self.best_epoch,
                        "lr": self._current_lr(),
                        "throughput": train_metrics.get("train/throughput", 0.0),
                    }
                    if val_metrics:
                        monitor_data["val_loss"] = val_metrics.get("val/total", 0.0)
                    
                    if torch.cuda.is_available():
                        try:
                            monitor_data["gpu_mem_GB"] = torch.cuda.memory_allocated(0) / 1024**3
                        except Exception:
                            pass
                    
                    live_mon.update(**monitor_data)

                    _prog.update(
                        _task,
                        advance=1,
                        description=f"[section]AE Training[/section]  best_val={self.best_val_total:.4f} @ ep {self.best_epoch}",
                    )

                    if not improved:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve >= self.cfg.early_stop_patience:
                            self.logger.subsection(f"[EarlyStopping] No improvement in {self.cfg.early_stop_patience} epochs — stopping at epoch {epoch}.")
                            break
                    else:
                        self.epochs_no_improve = 0

        self._save_checkpoint("final")
        self._plot_history()
        return self.history

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accum = MetricMeter()

        total_data_time    = 0.0
        total_compute_time = 0.0
        total_samples      = 0
        epoch_t0           = time.perf_counter()
        t_data_start       = time.perf_counter()

        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]AE Train[/section] - epoch {epoch}/{self.cfg.epochs}", total=len(self.train_loader))
            for profile_a, profile_b, _ in self.train_loader:
                data_time         = time.perf_counter() - t_data_start
                total_data_time  += data_time

                t_compute = time.perf_counter()
                profile_a = profile_a.to(self.device, non_blocking=True)
                profile_b = profile_b.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                    output_a = self.model(profile_a)
                    output_b = self.model(profile_b)
                    losses   = self.criterion(output_a, profile_a, output_b)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses["total"]).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                compute_time         = time.perf_counter() - t_compute
                total_compute_time  += compute_time

                bs = profile_a.shape[0]
                accum.update(losses, bs)
                total_samples += bs

                if self.writer is not None and (self.global_step % self.cfg.log_every_n_steps == 0):
                    self.writer.add_scalar("train_step/total",          float(losses["total"].item()), self.global_step)
                    self.writer.add_scalar("train_step/data_time_ms",   data_time    * 1e3,            self.global_step)
                    self.writer.add_scalar("train_step/compute_time_ms", compute_time * 1e3,           self.global_step)
                self.global_step += 1

                t_data_start = time.perf_counter()
                _prog.update(_task, advance=1, description=f"[section]AE Train[/section] - epoch {epoch}/{self.cfg.epochs}  loss={float(losses['total'].item()):.4f}")

        epoch_time = time.perf_counter() - epoch_t0
        throughput = total_samples / epoch_time if epoch_time > 0 else 0.0
        metrics    = accum.compute(prefix="train/")
        metrics["train/data_time_s"]    = total_data_time
        metrics["train/compute_time_s"] = total_compute_time
        metrics["train/throughput"]     = throughput
        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        accum = MetricMeter()
        with self.logger.track(transient=True) as _prog:
            _task = _prog.add_task(f"[section]AE Val[/section] - epoch {epoch}/{self.cfg.epochs}", total=len(self.val_loader))
            for profile_a, profile_b, _ in self.val_loader:
                profile_a = profile_a.to(self.device, non_blocking=True)
                profile_b = profile_b.to(self.device, non_blocking=True)
                output_a  = self.model(profile_a)
                output_b  = self.model(profile_b)
                losses    = self.criterion(output_a, profile_a, output_b)
                accum.update(losses, profile_a.shape[0])
                _prog.advance(_task)
        return accum.compute(prefix="val/")

    def _plot_history(self) -> None:
        if not self.history:
            return

        epochs = [r["epoch"] for r in self.history]
        component_names: set[str] = set()
        for record in self.history:
            for k in record:
                if k.startswith("train/"):
                    component_names.add(k[len("train/"):])

        loss_components = sorted(
            c for c in component_names
            if not any(c.endswith(s) for s in ("_time_s", "_time_ms", "throughput"))
        )

        series : dict[str, dict[str, list[float] | None]] = {}
        for component in loss_components:
            train_vals = [float(r.get(f"train/{component}", float("nan"))) for r in self.history]
            val_vals_raw = [r.get(f"val/{component}") for r in self.history]
            has_val = any(v is not None for v in val_vals_raw)
            val_vals = [float(v) if v is not None else float("nan") for v in val_vals_raw] if has_val else None
            series[component] = {"train": train_vals, "val": val_vals}
            self.plotter.plot_loss_component(component, epochs, train_vals, val_vals)
            self.logger.subsection(f"[LossPlot] -> images/loss_{component}.png")

        ordered = ["total"] + sorted(c for c in loss_components if c != "total")
        ordered = [c for c in ordered if c in series]
        if ordered:
            self.plotter.plot_loss_overview(ordered, epochs, series)
            self.logger.subsection("[LossPlot] -> images/loss_overview.png")

    def _save_checkpoint(self, tag: str) -> dict[str, Path]:
        full_path    = self.checkpoint_dir / f"autoencoder_{tag}.pt"
        encoder_path = self.checkpoint_dir / f"encoder_{tag}.pt"
        decoder_path = self.checkpoint_dir / f"decoder_{tag}.pt"

        torch.save({
            "model_state_dict"     : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "scheduler_state_dict" : self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict"    : self.scaler.state_dict(),
            "ae_config"            : self.ae_config,
            "history"              : self.history,
            "best_val_total"       : self.best_val_total,
            "best_epoch"           : self.best_epoch,
            "tag"                  : tag,
        }, full_path)

        torch.save({
            "encoder_state_dict"     : self.model.encoder.state_dict(),
            "projection_state_dict"  : self.model.projection_head.state_dict() if self.model.projection_head is not None else None,
            "ae_config"              : self.ae_config,
            "tag"                    : tag,
        }, encoder_path)

        torch.save({
            "decoder_state_dict" : self.model.decoder.state_dict(),
            "ae_config"          : self.ae_config,
            "tag"                : tag,
        }, decoder_path)

        self.logger.subsection(f"[Checkpoint] -> {full_path.name}, {encoder_path.name}, {decoder_path.name}")
        return {"full": full_path, "encoder": encoder_path, "decoder": decoder_path}

    def _update_best(self,
                     val_metrics   : dict[str, float],
                     train_metrics : dict[str, float],
                     epoch         : int) -> bool:
        key   = "val/total" if "val/total" in val_metrics else "train/total"
        score = val_metrics.get("val/total") if "val/total" in val_metrics else train_metrics.get("train/total", math.inf)
        if score < self.best_val_total:
            self.best_val_total = score
            self.best_epoch     = epoch
            self._save_checkpoint("best")
            self.logger.subsection(f"[Best] {key}={score:.6f} at epoch {epoch}")
            return True
        return False

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _metric(record: dict[str, float], key: str) -> float | None:
        value = record.get(key)
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _fmt_metric(value: float | None, precision: int = 6) -> str:
        if value is None:
            return "--"
        return f"{value:.{precision}f}"

    def _log_group(
        self,
        title       : str,
        train_items : list[tuple[str, str]],
        val_items   : list[tuple[str, str]] | None = None,
    ) -> None:
        from rich.table import Table
        tbl = Table(title=title, show_header=True, header_style="bold cyan", expand=False)
        tbl.add_column("Metric", style="key", no_wrap=True)
        tbl.add_column("Train",  style="value", justify="right")
        if val_items is not None:
            tbl.add_column("Val",  style="value", justify="right")
        val_map = dict(val_items) if val_items is not None else {}
        for label, value in train_items:
            row = [label, value]
            if val_items is not None:
                row.append(val_map.get(label, "--"))
            tbl.add_row(*row)
        self.logger.table(tbl)

    def _log_record(self, record: dict[str, float]) -> None:
        epoch = int(record["epoch"])
        lr    = self._fmt_metric(self._metric(record, "lr"), precision=8)
        self.logger.rule(f"Epoch {epoch:04d}  ·  lr={lr}", style="cyan")

        self._log_group(
            title="objective",
            train_items=[
                ("total", self._fmt_metric(self._metric(record, "train/total"))),
                ("recon", self._fmt_metric(self._metric(record, "train/reconstruction"))),
            ],
            val_items=[
                ("total", self._fmt_metric(self._metric(record, "val/total"))),
                ("recon", self._fmt_metric(self._metric(record, "val/reconstruction"))),
            ],
        )

        self._log_group(
            title="regularizers",
            train_items=[
                ("var", self._fmt_metric(self._metric(record, "train/variance"))),
                ("cov", self._fmt_metric(self._metric(record, "train/covariance"))),
                ("con", self._fmt_metric(self._metric(record, "train/contrastive"))),
            ],
            val_items=[
                ("var", self._fmt_metric(self._metric(record, "val/variance"))),
                ("cov", self._fmt_metric(self._metric(record, "val/covariance"))),
                ("con", self._fmt_metric(self._metric(record, "val/contrastive"))),
            ],
        )

        self._log_group(
            title="runtime",
            train_items=[
                ("throughput", self._fmt_metric(self._metric(record, "train/throughput"), precision=2)),
                ("data_s",     self._fmt_metric(self._metric(record, "train/data_time_s"), precision=3)),
                ("compute_s",  self._fmt_metric(self._metric(record, "train/compute_time_s"), precision=3)),
            ],
            val_items=None,
        )

        if self.writer is not None:
            for k, v in record.items():
                if k == "epoch":
                    continue
                self.writer.add_scalar(k, float(v), int(record["epoch"]))

    def _append_csv(self, record: dict[str, float]) -> None:
        write_header = not self._csv_header_written and not self._csv_path.exists()
        mode = "w" if write_header else "a"
        with open(self._csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header or not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(record)
