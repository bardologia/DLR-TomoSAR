import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from core.logger import ShapeLogger, ModelSummary, Tracker
from tqdm import tqdm
import gc
from pathlib import Path


GAUSSIAN_PARAM_NAMES = ["a1", "mu1", "sig1", "a2", "mu2", "sig2"]
NOISE_PARAM_NAME    = "log_noise_var"
PARAM_NAMES         = GAUSSIAN_PARAM_NAMES  # kept for backward compat


def reconstruct_double_gaussian(params: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct 2 superimposed Gaussian curves from predicted parameters.

    Args:
        params: (B, 6, H, W) — [a1, mu1, sigma1, a2, mu2, sigma2]
        x_axis: (N,) — discrete x values where curves are evaluated

    Returns:
        (B, N, H, W) — reconstructed curves
    """
    eps  = 1e-8
    x_axis = x_axis.to(params.device)

    a1   = params[:, 0:1, :, :]
    mu1  = params[:, 1:2, :, :]
    sig1 = params[:, 2:3, :, :]
    a2   = params[:, 3:4, :, :]
    mu2  = params[:, 4:5, :, :]
    sig2 = params[:, 5:6, :, :]

    x = x_axis.reshape(1, -1, 1, 1)                           # (1, N, 1, 1)

    g1 = a1 * torch.exp(-((x - mu1) ** 2) / (2 * sig1 ** 2 + eps))
    g2 = a2 * torch.exp(-((x - mu2) ** 2) / (2 * sig2 ** 2 + eps))

    return g1 + g2                                             # (B, N, H, W)


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config           = config
        self.logger           = logger
        self.tracker          = tracker
        self.patience         = self.config.early_stopping.patience
        self.min_delta        = self.config.early_stopping.min_delta
        self.restore_best     = self.config.early_stopping.restore_best
        
        self.logger.section("[Early Stopping]")
        self.logger.subsection(f"Patience       : {self.patience}")
        self.logger.subsection(f"Min Delta      : {self.min_delta}")
        self.logger.subsection(f"Restore Best   : {self.restore_best} \n")

        self.best_loss        = None
        self.counter          = 0
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_state(model)
            self.counter = 0
            stop = False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_state(model)
            self.tracker.log_scalar("early_stopping/best_val_loss", self.best_loss, epoch)
            stop = False

        else:
            self.counter += 1
            stop = (self.counter >= self.patience)

        self.tracker.log_scalar("early_stopping/counter", self.counter, epoch)

        if stop and self.restore_best:
            self.restore_model(model)

        return stop

    def _save_state(self, model):
        if self.restore_best:
            self.best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state, strict=True)


class Warmup:
    def __init__(self, optimizer, config, logger, tracker):
        self.config              = config
        self.logger              = logger
        self.tracker             = tracker
        self.optimizer           = optimizer
        self.warmup_steps        = self.config.warmup.warmup_steps
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.base_lrs            = [group['lr'] for group in optimizer.param_groups]
        self.current_step        = 0
        self.warmup_finished     = False
        self._logged_completion  = False

        self.logger.section("[Warmup Scheduler]")
        self.logger.subsection(f"Warmup Enabled      : {self.enabled}")
        self.logger.subsection(f"Warmup Steps        : {self.warmup_steps}")
        self.logger.subsection(f"Warmup Start Factor : {self.warmup_start_factor} \n")

        if self.enabled and self.warmup_steps > 0:
            self._apply_warmup_factor(self.warmup_start_factor)
    
    def _apply_warmup_factor(self, factor: float) -> None:
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor
        
    def step(self) -> None:
        if not self.enabled or self.warmup_steps <= 0:
            return

        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            self._apply_warmup_factor(factor)
            self.tracker.log_scalar("warmup/lr_factor", factor, self.current_step)
        
        elif not self.warmup_finished:
            self._apply_warmup_factor(1.0)
            self.warmup_finished = True
            
            if not self._logged_completion:
                self.logger.info(f"Warmup completed at step {self.current_step}. Learning rates restored to base values.")
                self.tracker.log_scalar("warmup/lr_factor", 1.0, self.current_step)
                self._logged_completion = True
    
    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0


class Scheduler:
    def __init__(self, optimizer, warmup, config, logger, tracker):
        self.config    = config
        self.optimizer = optimizer
        self.warmup    = warmup
        self.logger    = logger
        self.tracker   = tracker
        self.scheduler = CosineAnnealingLR(optimizer, T_max=self.config.scheduler.epochs, eta_min=self.config.scheduler.eta_min)
     
        self.logger.section("[Learning Rate Scheduler]")
        self.logger.subsection(f"Scheduler Type    : CosineAnnealingLR")
        self.logger.subsection(f"Scheduler Params  : T_max={self.config.scheduler.epochs}")
        self.logger.subsection(f"Scheduler Eta Min : {self.config.scheduler.eta_min}")
        self.logger.subsection(f"Warmup Enabled    : {self.warmup.enabled} \n")
        
    def step(self, epoch: int) -> None:
        if self.warmup and not self.warmup.is_finished():
            return
        
        self.scheduler.step()
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            self.tracker.log_scalar(f"scheduler/lr_{group_name}", param_group['lr'], epoch)
    
    def state_dict(self) -> dict:
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)


class EMA:
    def __init__(self, model: nn.Module, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker
        self.enabled = self.config.ema.use_ema
        self.decay   = self.config.ema.ema_decay
        
        self.logger.section("[Exponential Moving Average (EMA)]")
        self.logger.subsection(f"EMA Enabled: {self.enabled}")
        self.logger.subsection(f"EMA Decay  : {self.decay} \n")

        self.shadow  = {}
        self.backup  = {}
        
        if self.enabled:
            self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int = None) -> None:
        if not self.enabled:
            return
        
        total_divergence = 0.0
        shadow_norm      = 0.0
        model_norm       = 0.0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                self.logger.warning(f"EMA: Initialized shadow for new parameter '{name}'")
                continue
            
            divergence        = (self.shadow[name] - param.detach()).norm().item()
            total_divergence += divergence
            
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            
            shadow_norm += self.shadow[name].norm().item()
            model_norm  += param.norm().item()
        
        if step is not None:
            self.tracker.log_scalar("ema/param_divergence", total_divergence, step)
            self.tracker.log_scalar("ema/shadow_norm",      shadow_norm, step)
            self.tracker.log_scalar("ema/model_norm",       model_norm, step)
            self.tracker.log_scalar("ema/norm_ratio",       shadow_norm / (model_norm + 1e-8), step)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        
        self.backup = {}
        
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "enabled" : self.enabled,
            "decay"   : self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled = state['enabled']
        self.decay   = state['decay']
        self.shadow  = state['shadow']
        self.backup  = {}


class Checkpoint:
    def __init__(self, logger, tracker):
        self.logger = logger
        self.tracker = tracker
    
    def save(self, trainer, path: str, epoch: int) -> None:
        checkpoint = {
            "epoch"         : epoch,
            "global_step"   : trainer.global_step,
            "best_val_loss" : trainer.best_val_loss,
            "best_epoch"    : trainer.best_epoch,
            "best_metrics"  : trainer.best_metrics,
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
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.best_epoch    = checkpoint["best_epoch"]
        trainer.best_metrics  = checkpoint["best_metrics"]
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


class Loss:
    def __init__(self, x_axis, logger, tracker):
        self.x_axis  = x_axis
        self.logger  = logger
        self.tracker = tracker

        self.logger.section("[Loss Function]")
        self.logger.subsection(f"Curve reconstruction MSE (double Gaussian vs experimental)")
        self.logger.subsection(f"Heteroscedastic noise head: enabled when out_channels > 6")
        self.logger.subsection(f"X-axis sample points : {len(x_axis)} \n")

    def __call__(self, pred_params, exp_curves, epoch):
        gaussian_params = pred_params[:, :6]                                         # (B, 6, H, W)
        pred_curves     = reconstruct_double_gaussian(gaussian_params, self.x_axis)  # (B, N, H, W)
        mse_loss        = F.mse_loss(pred_curves, exp_curves)

        if pred_params.shape[1] > 6:
            # ── Heteroscedastic noise model ──────────────────────────
            # The 7th channel is log(σ) per pixel. We model:
            #   y ~ N(gaussians(θ), σ²)     →   NLL ∝ ||g-y||²/(2σ²) + log(σ)
            # This lets the Gaussians stay sharp; noisy pixels get high σ
            # and contribute less to the parameter gradients.
            log_sigma   = pred_params[:, 6:7, :, :]                        # (B, 1, H, W)
            sigma       = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
            sq_diff     = (pred_curves - exp_curves) ** 2                  # (B, N, H, W)
            nll         = 0.5 * sq_diff / (sigma ** 2 + 1e-8) + log_sigma  # broadcast over N
            noise_loss  = nll.mean()
            total_loss  = noise_loss

            self.tracker.log_dict("loss", {
                "log_total_loss"  : np.log1p(total_loss.item()),
                "log_mse_loss"    : np.log1p(mse_loss.item()),
                "log_noise_loss"  : np.log1p(noise_loss.item()),
                "noise_sigma_mean": sigma.mean().item(),
                "noise_sigma_std" : sigma.std().item(),
            }, epoch)

            return {
                "total_loss" : total_loss,
                "mse_loss"   : mse_loss,
                "noise_loss" : noise_loss,
                "sigma_mean" : sigma.mean().item(),
            }
        else:
            total_loss = mse_loss

            self.tracker.log_dict("loss", {
                "log_total_loss" : np.log1p(total_loss.item()),
                "log_mse_loss"   : np.log1p(mse_loss.item()),
            }, epoch)

            return {
                "total_loss": total_loss,
                "mse_loss": mse_loss,
            }


class Metrics:
    def __init__(self, verbose: bool, tracker, logger, x_axis):
        self.verbose = verbose
        self.tracker = tracker
        self.logger  = logger
        self.x_axis  = x_axis

    def track_results(self, results: dict, epoch: int, stage: str = "validation"):
        self.tracker.log_dict(f"{stage}_curve_fit", {
            "curve_mse"  : float(results["curve_mse"]),
            "curve_mae"  : float(results["curve_mae"]),
            "curve_rmse" : float(results["curve_rmse"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_pixel_r2", {
            "r2_mean"   : float(results["pixel_r2_mean"]),
            "r2_std"    : float(results["pixel_r2_std"]),
            "r2_median" : float(results["pixel_r2_median"]),
            "r2_min"    : float(results["pixel_r2_min"]),
            "r2_max"    : float(results["pixel_r2_max"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_cosine_similarity", {
            "cos_sim_mean"   : float(results["cos_sim_mean"]),
            "cos_sim_std"    : float(results["cos_sim_std"]),
            "cos_sim_median" : float(results["cos_sim_median"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_pixel_mse_dist", {
            "mean"   : float(results["pixel_mse_mean"]),
            "std"    : float(results["pixel_mse_std"]),
            "median" : float(results["pixel_mse_median"]),
            "max"    : float(results["pixel_mse_max"]),
            "min"    : float(results["pixel_mse_min"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_pixel_mae_dist", {
            "mean"   : float(results["pixel_mae_mean"]),
            "std"    : float(results["pixel_mae_std"]),
            "median" : float(results["pixel_mae_median"]),
            "max"    : float(results["pixel_mae_max"]),
            "min"    : float(results["pixel_mae_min"]),
        }, epoch)

        param_stats = {name: float(results[f"{name}_mean"]) for name in GAUSSIAN_PARAM_NAMES}
        self.tracker.log_dict(f"{stage}_param_means", param_stats, epoch)

        if f"{NOISE_PARAM_NAME}_mean" in results:
            self.tracker.log_dict(f"{stage}_noise", {
                "log_sigma_mean" : float(results[f"{NOISE_PARAM_NAME}_mean"]),
                "sigma_mean"     : float(results["sigma_mean"]),
                "sigma_std"      : float(results["sigma_std"]),
                "sigma_min"      : float(results["sigma_min"]),
                "sigma_max"      : float(results["sigma_max"]),
            }, epoch)

    def calculate(self, epoch, pred_params, exp_curves, stage="validation"):
        # pred_params: (B, 6+, H, W)  —  predicted Gaussian params [+ optional noise channel]
        # exp_curves:  (B, N, H, W)   —  experimental curves
        gaussian_params = pred_params[:, :6]  # always first 6 channels
        pred_curves     = reconstruct_double_gaussian(gaussian_params, self.x_axis)
        diff            = pred_curves - exp_curves
        abs_diff        = torch.abs(diff)

        # ── Per-pixel curve MSE / MAE  (average over spectral dim N) ──
        pixel_mse = (diff ** 2).mean(dim=1)           # (B, H, W)
        pixel_mae = abs_diff.mean(dim=1)              # (B, H, W)

        # ── Per-pixel R²  (how well each pixel's curve is fitted) ──
        ss_res   = (diff ** 2).sum(dim=1)
        exp_mean = exp_curves.mean(dim=1, keepdim=True)
        ss_tot   = ((exp_curves - exp_mean) ** 2).sum(dim=1)
        pixel_r2 = 1 - ss_res / (ss_tot + 1e-8)      # (B, H, W)

        # ── Per-pixel cosine similarity ──
        cos_sim = F.cosine_similarity(pred_curves, exp_curves, dim=1)  # (B, H, W)

        # ── Overall curve-level metrics ──
        curve_mse  = pixel_mse.mean().item()
        curve_mae  = pixel_mae.mean().item()
        curve_rmse = float(np.sqrt(curve_mse))

        overall_ss_res = torch.sum(diff ** 2).item()
        overall_ss_tot = torch.sum((exp_curves - exp_curves.mean()) ** 2).item()
        overall_r2     = 1 - (overall_ss_res / overall_ss_tot) if overall_ss_tot > 0 else 0

        # ── Predicted-parameter distribution (no ground-truth params needed) ──
        param_stats = {}
        for i, name in enumerate(GAUSSIAN_PARAM_NAMES):
            p = gaussian_params[:, i]
            param_stats[f"{name}_mean"] = p.mean().item()
            param_stats[f"{name}_std"]  = p.std().item()
            param_stats[f"{name}_min"]  = p.min().item()
            param_stats[f"{name}_max"]  = p.max().item()

        # ── Noise head statistics (if model predicts noise channel) ──
        if pred_params.shape[1] > 6:
            log_sigma = pred_params[:, 6]
            sigma     = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
            param_stats[f"{NOISE_PARAM_NAME}_mean"] = log_sigma.mean().item()
            param_stats[f"{NOISE_PARAM_NAME}_std"]  = log_sigma.std().item()
            param_stats["sigma_mean"]                = sigma.mean().item()
            param_stats["sigma_std"]                 = sigma.std().item()
            param_stats["sigma_min"]                 = sigma.min().item()
            param_stats["sigma_max"]                 = sigma.max().item()

        if self.verbose:
            self.logger.section(f"[Epoch {epoch}] Curve Fitting — {stage.capitalize()}")
            self.logger.subsection(f"Curve : MSE={curve_mse:.6f}  MAE={curve_mae:.6f}  RMSE={curve_rmse:.6f}")
            self.logger.subsection(f"R²    : Overall={overall_r2:.4f}  Pixel Mean={pixel_r2.mean():.4f}  Median={pixel_r2.median():.4f}  Min={pixel_r2.min():.4f}")
            self.logger.subsection(f"CosSim: Mean={cos_sim.mean():.4f}  Median={cos_sim.median():.4f}")
            for name in GAUSSIAN_PARAM_NAMES:
                self.logger.subsection(
                    f"  {name}: mean={param_stats[f'{name}_mean']:.4f}  std={param_stats[f'{name}_std']:.4f}  "
                    f"min={param_stats[f'{name}_min']:.4f}  max={param_stats[f'{name}_max']:.4f}"
                )
            if pred_params.shape[1] > 6:
                self.logger.subsection(
                    f"  Noise σ: mean={param_stats['sigma_mean']:.4f}  std={param_stats['sigma_std']:.4f}  "
                    f"min={param_stats['sigma_min']:.4f}  max={param_stats['sigma_max']:.4f}"
                )
            self.logger.subsection("")

        results = {
            "curve_mse"  : curve_mse,
            "curve_mae"  : curve_mae,
            "curve_rmse" : curve_rmse,
            "overall_r2" : overall_r2,

            "pixel_r2_mean"   : pixel_r2.mean().item(),
            "pixel_r2_std"    : pixel_r2.std().item(),
            "pixel_r2_median" : pixel_r2.median().item(),
            "pixel_r2_min"    : pixel_r2.min().item(),
            "pixel_r2_max"    : pixel_r2.max().item(),

            "cos_sim_mean"   : cos_sim.mean().item(),
            "cos_sim_std"    : cos_sim.std().item(),
            "cos_sim_median" : cos_sim.median().item(),

            "pixel_mse_mean"   : pixel_mse.mean().item(),
            "pixel_mse_std"    : pixel_mse.std().item(),
            "pixel_mse_median" : pixel_mse.median().item(),
            "pixel_mse_max"    : pixel_mse.max().item(),
            "pixel_mse_min"    : pixel_mse.min().item(),

            "pixel_mae_mean"   : pixel_mae.mean().item(),
            "pixel_mae_std"    : pixel_mae.std().item(),
            "pixel_mae_median" : pixel_mae.median().item(),
            "pixel_mae_max"    : pixel_mae.max().item(),
            "pixel_mae_min"    : pixel_mae.min().item(),
        }
        results.update(param_stats)

        self.track_results(results, epoch, stage)
        return results


class Trainer:
    def __init__(self, model, x_axis, config, run_dir, logger):
        self.logger = logger
        self.config = config
        
        self.logger.section("[Training Start]")
        self.logger.subsection(f"Device Name   : {torch.cuda.get_device_name(0)}")
        self.logger.subsection(f"Total Memory  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.subsection(f"CUDA Version  : {torch.version.cuda}")
        self.logger.subsection(f"Log Directory : {self.config.io.logdir} \n")
        
        self.checkpoint_path = Path(run_dir) / "best_model.pt"
        self.tracker         = Tracker(writer=self.config.io.writer)
        
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
        self.criterion      = Loss(self.x_axis, self.logger, self.tracker)
        self.metrics        = Metrics(self.config.training.verbose, self.tracker, logger=self.logger, x_axis=self.x_axis)
        self.checkpoint     = Checkpoint(self.logger, self.tracker)
        self.shape_logger   = ShapeLogger(model = self.model, logger = self.logger).attach()
        self.summary        = ModelSummary(logger = self.logger, model = self.model)
        self.summary.run()
        self.summary.save_markdown(os.path.join(self.config.io.logdir, "model_summary.md"), title="Model Summary")
        
        self.best_val_loss = float("inf")
        self.best_epoch    = -1
        self.best_metrics  = {}
        self.global_step   = 0
        self.train_losses  = []
        self.val_losses    = []
              
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def _get_head_params(self):
        """Collect output-head parameters across all model variants."""
        head_params = []
        if hasattr(self.model, 'output_head'):
            head_params += list(self.model.output_head.parameters())
        if hasattr(self.model, 'output_heads'):
            head_params += list(self.model.output_heads.parameters())
        if hasattr(self.model, 'score_final'):
            head_params += list(self.model.score_final.parameters())
        if hasattr(self.model, 'score_pool4'):
            head_params += list(self.model.score_pool4.parameters())
        if hasattr(self.model, 'score_pool3'):
            head_params += list(self.model.score_pool3.parameters())
        return head_params

    def make_param_groups(self):
        param_groups = []

        head_params     = self._get_head_params()
        head_param_ids  = set(id(p) for p in head_params)
        backbone_params = [p for p in self.model.parameters() if id(p) not in head_param_ids]

        param_groups.append({
            'params'       : backbone_params,
            'lr'           : self.config.optimizer.lr_backbone,
            'weight_decay' : self.config.optimizer.weight_decay_backbone,
            'name'         : 'backbone'
        })

        param_groups.append({
            'params'       : head_params,
            'lr'           : self.config.optimizer.lr_output_head,
            'weight_decay' : self.config.optimizer.weight_decay_output_head,
            'name'         : 'output_head'
        })

        self.logger.section("[Optimizer Parameter Groups]")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            self.logger.subsection(
                f"{group['name']} - LR: {group['lr']}, Weight Decay: {group['weight_decay']}, Parameters: {num_params:,}"
            )

        return param_groups
    
    def forward(self, images, exp_curves, epoch):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_params = self.model(images)
            loss_dict   = self.criterion(pred_params, exp_curves, epoch)
            self.shape_logger.detach()
        
        return pred_params, loss_dict
    
    def backward(self, loss, step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
            if step:
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                if self.global_step % 100 == 0:
                    self.tracker.log_gradients(self.model, self.global_step, max_grad_norm=self.config.training.max_grad_norm)
                    self.tracker.log_optimizer(self.optimizer, self.global_step)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.warmup.step()
                self.global_step += 1
                self.ema.update(self.model, step=self.global_step)
        else:
            loss.backward()
            if step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                if self.global_step % 100 == 0:
                    self.tracker.log_gradients(self.model, self.global_step, max_grad_norm=self.config.training.max_grad_norm)
                    self.tracker.log_optimizer(self.optimizer, self.global_step)
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.warmup.step()
                self.global_step += 1
                self.ema.update(self.model, step=self.global_step)

    def train_epoch(self, train_loader, epoch, loader_len=None):
        self.model.train()
        total_loss   = 0
        num_batches  = 0
        batch_losses = []
                
        activation_hooks = []
        if epoch > 0 and epoch % 10 == 0:
            activation_hooks = self.tracker.log_activations(self.model, epoch)
    
        try:
            for batch_idx, (images, exp_curves) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", leave=False, total=len(train_loader)):
                images     = images.to(self.device, non_blocking=True)
                exp_curves = exp_curves.to(self.device, non_blocking=True)

                pred_params, loss_dict = self.forward(images, exp_curves, epoch)
                loss = loss_dict["total_loss"] / self.accumulation_steps

                should_step = (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) >= len(train_loader)
                self.backward(loss, step=should_step)
                
                loss_value        = loss.item()
                total_loss       += loss_value * self.accumulation_steps
                num_batches      += 1
                batch_losses.append(loss_value * self.accumulation_steps)
        finally:
            for hook in activation_hooks:
                hook.remove()

        if epoch > 0 and epoch % 10 == 0:
            self.tracker.log_weights(self.model, epoch)
                
    def evaluate(self, loader, epoch, stage="validation"):
        self.model.eval()
        self.ema.apply_to(self.model)

        try:
            total_loss      = 0
            all_pred_params = []
            all_exp_curves  = []
            num_batches     = 0
                        
            with torch.no_grad():
                for (images, exp_curves) in tqdm(loader, desc=f"Evaluating {stage} Epoch {epoch+1}/{self.epochs}", leave=False, total=len(loader)):
                    images     = images.to(self.device, non_blocking=True)
                    exp_curves = exp_curves.to(self.device, non_blocking=True)
                    pred_params, loss_dict = self.forward(images, exp_curves, epoch)
                    total_loss += loss_dict["total_loss"].item()
                    all_pred_params.append(pred_params.cpu())
                    all_exp_curves.append(exp_curves.cpu())
                    num_batches += 1
                    
            avg_loss    = total_loss / max(1, num_batches)
            pred_params = torch.cat(all_pred_params)
            exp_curves  = torch.cat(all_exp_curves)

            metrics = self.metrics.calculate(epoch, pred_params, exp_curves, stage=stage)
        finally:
            self.ema.restore(self.model)

        results = {
            "avg_loss"    : avg_loss,
            "pred_params" : pred_params,
            "exp_curves"  : exp_curves,
            "metrics"     : metrics,
        }

        return results

    def train(self, train_loader, val_loader, test_loader):    
        self.logger.section("[Training Loop]")
        self.logger.subsection(f"Train loader size      = {len(train_loader)}")
        self.logger.subsection(f"Validation loader size = {len(val_loader)}")
        self.logger.subsection(f"Test loader size       = {len(test_loader)}")
        
        self._clear_memory()
        self.optimizer.zero_grad()
     
        train_loader_len = len(train_loader)
        if self.config.overfit.enabled:
            single_batch      = next(iter(train_loader))
            data_loader       = [single_batch] * train_loader_len
            eval_train_loader = [single_batch]
            self.logger.warning(f"Overfitting mode enabled: training on a single batch repeated {train_loader_len} times.")
        else:
            data_loader       = train_loader
            eval_train_loader = train_loader
        
        epochs = self.epochs
        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_num = epoch + 1
            
            self.train_epoch(data_loader, epoch, train_loader_len)
     
            val_results   = self.evaluate(val_loader, epoch, stage="validation")
            self.logger.info(f"Epoch {epoch} - Validation Loss: {val_results['avg_loss']:.4f}")
            self.train_losses.append(val_results["avg_loss"])
            
            train_results = self.evaluate(eval_train_loader, epoch, stage="train")
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_results['avg_loss']:.4f}")
            self.val_losses.append(val_results["avg_loss"])

            self._clear_memory()

            loss_comparison = {
                "train_loss": train_results['avg_loss'],
                "val_loss"  : val_results["avg_loss"],
            }
            
            self.tracker.log_dict("loss_comparison", loss_comparison, epoch)

            if val_results["avg_loss"] < self.best_val_loss:
                self.best_val_loss = float(val_results["avg_loss"])
                self.best_epoch    = int(epoch_num)
                self.best_metrics  = dict(val_results["metrics"])
                self.checkpoint.save(self, self.checkpoint_path, epoch_num)
                self.logger.subsection(f"New best model found at epoch {epoch_num} with validation loss {self.best_val_loss:.4f}")
        
            self.lr_scheduler.step(epoch)
            if self.early_stopping(val_results["avg_loss"], self.model, epoch):
                self.logger.warning(f"Early stopping triggered at epoch {epoch_num}. Best model from epoch {self.best_epoch} restored.")
                break

        self.shape_logger.save_markdown(path=Path(self.config.io.logdir) / "tensor_shape.md", sort_by_layer=True)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        train_final_results      = self.evaluate(train_loader, checkpoint["epoch"], stage="final_train")
        validation_final_results = self.evaluate(val_loader,   checkpoint["epoch"], stage="final_validation")
        test_final_results       = self.evaluate(test_loader,  checkpoint["epoch"], stage="final_test")
        
        return train_final_results, validation_final_results, test_final_results

 