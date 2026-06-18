from __future__ import annotations

from dataclasses import dataclass, field


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
    presence_bce      : float = 1.000000


@dataclass
class LossConfig:
    use_mse_curve    : bool  = False
    weight_mse_curve : float = 1.0

    use_l1_curve    : bool  = False
    weight_l1_curve : float = 0.0

    use_huber_curve    : bool  = False
    weight_huber_curve : float = 0.0
    huber_delta        : float = 1.0

    use_charbonnier_curve    : bool  = False
    weight_charbonnier_curve : float = 0.0
    charbonnier_eps          : float = 1e-3

    use_cosine_curve    : bool  = False
    weight_cosine_curve : float = 0.0

    use_spectral_coherence : bool  = False
    weight_spectral_coh    : float = 0.0
    spectral_coh_window    : int   = 7

    use_ssim_curve    : bool  = False
    weight_ssim_curve : float = 0.0
    ssim_window_size  : int   = 11
    ssim_sigma        : float = 1.5
    ssim_data_range   : float = 1.0
    ssim_k1           : float = 0.01
    ssim_k2           : float = 0.03
    ssim_axis         : str   = "elevation"

    use_param_l1    : bool  = False
    weight_param_l1 : float = 0.1

    use_param_huber    : bool  = False
    weight_param_huber : float = 0.0
    param_huber_delta  : float = 0.5

    param_weights : tuple = (1.0, 1.0, 1.0)
    param_match   : str   = "sort_gt_by_mu"

    use_active_normalization : bool  = False
    presence_balance         : bool  = False
    active_weight            : float = 1.0
    inactive_weight          : float = 1.0
    amp_focal_gamma          : float = 0.0
    amp_focal_delta          : float = 0.5

    use_presence_bce     : bool  = False
    weight_presence_bce  : float = 0.0
    presence_bce_balance : bool  = True
    presence_gate_thr    : float = 0.5

    amp_zero_thr       : float = 1e-3
    amp_zero_thr_torch : float = 1e-7

    use_smoothness_tv    : bool  = False
    weight_smoothness_tv : float = 1e-4

    use_total_power    : bool  = False
    weight_total_power : float = 0.0

    use_moments     : bool  = False
    weight_moments  : float = 0.0
    moments_weights : tuple = (1.0, 1.0, 1.0)

    use_coherence_resyn    : bool  = False
    weight_coherence_resyn : float = 0.0

    use_covariance_match    : bool  = False
    weight_covariance_match : float = 0.0

    use_capon_cycle    : bool  = False
    weight_capon_cycle : float = 0.0
    capon_loading      : float = 1e-2

    physics_floor            : float = 1e-3

    norm : LossNormalizationConfig = field(default_factory=LossNormalizationConfig)

    def eff(self, weight_key: str) -> float:
        alpha       = getattr(self, weight_key)
        norm_factor = getattr(self.norm, weight_key.removeprefix("weight_"), 1.0)
        return alpha * norm_factor


@dataclass
class LossCurriculumConfig:
    enabled    : bool = False
    swap_epoch : int  = 0

    warmup   : LossConfig = field(default_factory=LossConfig)
    complete : LossConfig = field(default_factory=LossConfig)

    reset_early_stopping : bool = False
    reset_lr             : bool = False
    reset_warmup         : bool = False
    reset_optimizer      : bool = False
