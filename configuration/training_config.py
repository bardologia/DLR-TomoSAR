from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class LossNormalizationConfig:
    mse_curve         : float = 0.256520
    l1_curve          : float = 0.799401
    huber_curve       : float = 1.304963
    charbonnier_curve : float = 0.795284
    cosine_curve      : float = 0.122937
    spectral_coh      : float = 0.117614
    ssim_curve        : float = 2.410647
    param_l1          : float = 1.000000
    param_huber       : float = 5.399934
    smoothness_tv     : float = 1.532997
    total_power       : float = 1.000000
    moments           : float = 1.000000
    coherence_resyn   : float = 1.000000
    covariance_match  : float = 1.000000
    capon_cycle       : float = 1.000000


@dataclass
class GeometryConfig:
    wavelength  : float = 0.23
    slant_range : float = 5000.0
    baselines   : tuple = (0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0)
    kz_values   : tuple = ()


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
    ssim_axis                : str   = "elevation"   

    use_param_l1             : bool  = False
    weight_param_l1          : float = 0.1

    use_param_huber          : bool  = False
    weight_param_huber       : float = 0.0
    param_huber_delta        : float = 0.5

    param_weights            : tuple = (1.0, 1.0, 1.0)
    param_match              : str   = "sort_gt_by_mu"   

    amp_zero_thr             : float = 1e-3
    amp_zero_thr_torch       : float = 1e-7

    use_smoothness_tv        : bool  = False
    weight_smoothness_tv     : float = 1e-4

    use_total_power          : bool  = False
    weight_total_power       : float = 0.0

    use_moments              : bool  = False
    weight_moments           : float = 0.0
    moments_weights          : tuple = (1.0, 1.0, 1.0)

    use_coherence_resyn      : bool  = False
    weight_coherence_resyn   : float = 0.0

    use_covariance_match     : bool  = False
    weight_covariance_match  : float = 0.0

    use_capon_cycle          : bool  = False
    weight_capon_cycle       : float = 0.0
    capon_loading            : float = 1e-2

    physics_floor            : float = 1e-3

    norm : LossNormalizationConfig = field(default_factory=LossNormalizationConfig)

    def eff(self, weight_key: str) -> float:
        alpha       = getattr(self, weight_key)
        norm_factor = getattr(self.norm, weight_key.removeprefix("weight_"), 1.0)
        return alpha * norm_factor


@dataclass
class LossCurriculumConfig:
    enabled              : bool       = False
    swap_epoch           : int        = 0

    warmup               : LossConfig = field(default_factory=LossConfig)
    complete             : LossConfig = field(default_factory=LossConfig)

    reset_early_stopping : bool       = False
    reset_lr             : bool       = False
    reset_warmup         : bool       = False
    reset_optimizer      : bool       = False


@dataclass
class GaussianConfig:
    n_default_gaussians : int
    x_min               : float
    x_max               : float
    amp_max             : float = 1000
    params_per_gaussian : int   = 3

    @classmethod
    def from_dataset(cls, dataset_dir: str | Path, n_gaussians: int) -> "GaussianConfig":
        meta_dir     = Path(dataset_dir) / "meta"
        candidates   = sorted(meta_dir.glob("config_state_*.json"))
        cfg          = json.loads(candidates[0].read_text())
        height_range = cfg["output_configs"]["height_range"]
        
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
    warmup_mode         : str   = "linear"   
    warmup_poly_power   : float = 2.0    


@dataclass
class SchedulerConfig:
    type         : str   = "cosine_annealing"
    epochs       : int   = 100
    eta_min      : float = 1e-6
    step_size    : int   = 30
    gamma        : float = 0.1
    milestones   : list  = field(default_factory=lambda: [30, 60, 90])
    start_factor : float = 1.0
    end_factor   : float = 0.1
    total_iters  : int   = 100
    power        : float = 1.0
    factor       : float = 0.1
    patience     : int   = 10
    threshold    : float = 1e-4
    T_0          : int   = 10
    T_mult       : float = 1.0


@dataclass
class EMAConfig:
    use_ema               : bool  = True
    ema_decay             : float = 0.999
    update_every_n_steps  : int   = 10


@dataclass
class IOConfig:
    logdir     : str = "/ste/rnd/User/vice_vi/DLR-TomoSAR/logs"
    tb_dir     : str = ""
    docs_dir   : str = ""
    logs_dir   : str = ""
    writer = None


@dataclass
class OptimizerConfig:
    lr           : float = 1e-3
    betas        : tuple = (0.9, 0.999)
    eps          : float = 1e-8
    weight_decay : float = 0.1


@dataclass
class OverfitConfig:
    enabled        : bool  = False
    max_steps      : int   = 50
    stop_threshold : float = 1e-6
    batch_size     : int   = 1


@dataclass
class TrainingConfigInner:
    device                      : str   = "cuda" if torch.cuda.is_available() else "cpu"
    epochs                      : int   = 3
    validation_frequency        : int   = 5
    use_amp                     : bool  = False
    gradient_accumulation_steps : int   = 1
    max_grad_norm               : float = None
    verbose                     : bool  = True
    log_debug                   : bool  = True
    log_all_losses              : bool  = False


@dataclass
class MemoryConfig:
    streaming_eval            : bool = True   
    eval_keep_pixel_arrays    : bool = True   
    eval_pixel_subsample      : int  = 0      
    clear_cache_every_n_steps : int  = 0      
    clear_cache_after_eval    : bool = False  
    clear_cache_after_epoch   : bool = False 


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
class GradientClipperConfig:
    clip_mode            : str   = "fixed"   # disabled | fixed | adaptive_percentile | adaptive_mean_std
    max_grad_norm        : float = 1.0        
    adaptive_window      : int   = 200        
    adaptive_percentile  : float = 95.0      
    adaptive_mean_std_k  : float = 2.0        
    clip_epsilon         : float = 1e-6      
    log_histogram_freq   : int   = 100       


@dataclass
class PermutationMetricsConfig:
    enabled          : bool  = True
    amp_threshold    : float = 1e-3
    max_G_for_margin : int   = 8


@dataclass
class TrainerConfig:
    gaussian            : GaussianConfig
    geometry            : GeometryConfig           = field(default_factory=GeometryConfig)
    early_stopping      : EarlyStoppingConfig      = field(default_factory=EarlyStoppingConfig)
    warmup              : WarmupConfig             = field(default_factory=WarmupConfig)
    scheduler           : SchedulerConfig          = field(default_factory=SchedulerConfig)
    ema                 : EMAConfig                = field(default_factory=EMAConfig)
    io                  : IOConfig                 = field(default_factory=IOConfig)
    optimizer           : OptimizerConfig          = field(default_factory=OptimizerConfig)
    training            : TrainingConfigInner      = field(default_factory=TrainingConfigInner)
    overfit             : OverfitConfig            = field(default_factory=OverfitConfig)
    curriculum          : LossCurriculumConfig     = field(default_factory=LossCurriculumConfig)
    resources           : ResourceConfig           = field(default_factory=ResourceConfig)
    memory              : MemoryConfig             = field(default_factory=MemoryConfig)
    gradient_clipper    : GradientClipperConfig    = field(default_factory=GradientClipperConfig)
    permutation_metrics : PermutationMetricsConfig = field(default_factory=PermutationMetricsConfig)
