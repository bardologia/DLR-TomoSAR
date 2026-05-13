from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class LossConfig:
    use_mse_curve            : bool  = False
    weight_mse_curve         : float = 1.0

    use_l1_curve             : bool  = False
    weight_l1_curve          : float = 0.0

    use_huber_curve          : bool  = False
    weight_huber_curve       : float = 0.0
    huber_delta              : float = 1.0

    use_charbonnier_curve    : bool  = False
    weight_charbonnier_curve : float = 0.0
    charbonnier_eps          : float = 1e-3

    use_cosine_curve         : bool  = False
    weight_cosine_curve      : float = 0.0


    use_spectral_coherence   : bool  = False
    weight_spectral_coh      : float = 0.0
    spectral_coh_window      : int   = 7

    use_ssim_curve           : bool  = False
    weight_ssim_curve        : float = 0.0
    ssim_window_size         : int   = 11
    ssim_sigma               : float = 1.5
    ssim_data_range          : float = 1.0
    ssim_k1                  : float = 0.01
    ssim_k2                  : float = 0.03
    ssim_axis                : str   = "elevation"   # "elevation" | "azimuth" | "range"

    use_param_l1             : bool  = False
    weight_param_l1          : float = 0.1

    use_param_huber          : bool  = False
    weight_param_huber       : float = 0.0
    param_huber_delta        : float = 0.5

    param_weights            : tuple = (1.0, 1.0, 1.0)
    param_match              : str   = "sorted_mu"

    use_smoothness_tv        : bool  = False
    weight_smoothness_tv     : float = 1e-4

    log_components_every     : int   = 1


@dataclass
class GaussianConfig:
    n_default_gaussians : int
    x_min               : float
    x_max               : float
    params_per_gaussian : int   = 3

    @classmethod
    def from_dataset(cls, dataset_dir: str | Path) -> "GaussianConfig":
        meta_dir   = Path(dataset_dir) / "meta"
        candidates = sorted(meta_dir.glob("config_state_*.json"))
        cfg = json.loads(candidates[0].read_text())
        height_range = cfg["output_configs"]["height_range"]
        params_dir   = Path(dataset_dir) / "params"
        param_meta   = json.loads(sorted(params_dir.glob("*/param_extraction_meta.json"))[0].read_text())
        n_gaussians  = param_meta["number_of_gaussians"]

        return cls(
            n_default_gaussians = n_gaussians,
            x_min               = float(height_range[0]),
            x_max               = float(height_range[1]),
        )

    def make_param_names(self, n_gaussians: int | None = None) -> list[str]:
        k = n_gaussians
        return [f"{prefix}{i + 1}" for i in range(k) for prefix in ("a", "mu", "sig")]

    @property
    def default_param_names(self) -> list[str]:
        return self.make_param_names(self.n_default_gaussians)


@dataclass
class EarlyStoppingConfig:
    patience     : int   = 15
    min_delta    : float = 0.001
    restore_best : bool  = True


@dataclass
class WarmupConfig:
    warmup_steps        : int   = 200
    warmup_start_factor : float = 0.1
    warmup_enabled      : bool  = True


@dataclass
class SchedulerConfig:
    epochs  : int   = 100
    eta_min : float = 1e-6


@dataclass
class EMAConfig:
    use_ema   : bool  = True
    ema_decay : float = 0.999


@dataclass
class IOConfig:
    logdir     : str = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs"
    tb_dir     : str = ""
    docs_dir   : str = ""
    logs_dir   : str = ""
    images_dir : str = ""
    writer = None


@dataclass
class OptimizerConfig:
    betas : tuple = (0.9, 0.999)
    eps   : float = 1e-8


@dataclass
class TrainingConfigInner:
    device                      : str   = "cuda" if torch.cuda.is_available() else "cpu"
    epochs                      : int   = 3
    validation_frequency        : int   = 5
    use_amp                     : bool  = False
    gradient_accumulation_steps : int   = 1
    max_grad_norm               : float = None
    verbose                     : bool  = True
    overfit_enabled             : bool  = False
    deep_validation             : bool  = True
    eval_train_split            : bool  = False
    log_debug                   : bool  = True   


@dataclass
class MemoryConfig:
    streaming_eval               : bool = True   
    eval_keep_pixel_arrays       : bool = True   
    eval_pixel_subsample         : int  = 0      
    clear_cache_every_n_steps    : int  = 0      
    clear_cache_after_eval       : bool = True   
    clear_cache_after_epoch      : bool = True   


@dataclass
class ResourceConfig:
    enabled            : bool  = True
    poll_interval_sec  : float = 5.0
    log_to_tensorboard : bool  = True
    log_to_csv         : bool  = True
    csv_path           : str   = ""        
    logs_dir           : str   = ""       
    warn_ram_pct       : float = 90.0
    warn_vram_pct      : float = 90.0
    warn_swap_pct      : float = 50.0
    warn_shm_pct       : float = 80.0
    warn_cooldown_sec  : float = 30.0


@dataclass
class TrainerConfig:
    gaussian       : GaussianConfig
    early_stopping : EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    warmup         : WarmupConfig        = field(default_factory=WarmupConfig)
    scheduler      : SchedulerConfig     = field(default_factory=SchedulerConfig)
    ema            : EMAConfig           = field(default_factory=EMAConfig)
    io             : IOConfig            = field(default_factory=IOConfig)
    optimizer      : OptimizerConfig     = field(default_factory=OptimizerConfig)
    training       : TrainingConfigInner = field(default_factory=TrainingConfigInner)
    loss           : LossConfig          = field(default_factory=LossConfig)
    resources      : ResourceConfig      = field(default_factory=ResourceConfig)
    memory         : MemoryConfig        = field(default_factory=MemoryConfig)
