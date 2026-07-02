from __future__ import annotations

from configuration.training.general.loss import ParamMatching


class AblationCatalog:

    WARMUP_PREFIX   = "curriculum.warmup."
    COMPLETE_PREFIX = "curriculum.complete."

    CURRICULUM_SWAP_EPOCH = 15

    PARAM_MATCH_BASELINE = ParamMatching.HUNGARIAN
    PARAM_MATCH_FULL     = ParamMatching.SORTED_GT

    GROUP_LR_DEFAULTS = (
        ("encoder_lr",     3e-4),
        ("bottleneck_lr",  3e-4),
        ("decoder_lr",     3e-4),
        ("output_head_lr", 1e-3),
    )
    SINGLE_GROUP_LR = 3e-4

    FULL_ARCHITECTURE     = "resunet"
    BASELINE_ARCHITECTURE = "unet"

    CHANNEL_NORMS = (
        ("out_amp",   "out_amp",   "robust_iqr_log1p", "zscore"),
        ("out_sigma", "out_sigma", "robust_iqr_log1p", "zscore"),
        ("pass_mag",  "pass_mag",  "robust_iqr_log1p", "zscore_log1p"),
        ("ifg_phase", "ifg_phase", "zscore",           "fixed_div_pi"),
        ("ifg_mag",   "ifg_mag",   "robust_iqr_log1p", "zscore"),
        ("dem",       "dem",       "robust_iqr_log1p", "zscore"),
    )

    AUGMENTATION_DEFAULTS = (
        ("p_flip_h", 0.5),
        ("p_flip_v", 0.5),
    )

    PHYSICS_TERMS = (
        ("coherence_resyn",  "use_coherence_resyn",  "weight_coherence_resyn",  0.05),
        ("covariance_match", "use_covariance_match", "weight_covariance_match", 0.05),
    )

    LOSS_TERMS = (
        ("spectral_coherence", "use_spectral_coherence", "weight_spectral_coh",  0.05),
        ("ssim",               "use_ssim_curve",         "weight_ssim_curve",    0.05),
        ("smoothness_tv",      "use_smoothness_tv",      "weight_smoothness_tv", 1e-4),
    )

    @classmethod
    def _channel_norm_features(cls) -> list[dict]:
        return [
            {
                "label"   : label,
                "group"   : "channel normalization",
                "enable"  : {f"normalization.{field}": full},
                "degrade" : {f"normalization.{field}": degraded},
            }
            for label, field, full, degraded in cls.CHANNEL_NORMS
        ]

    @classmethod
    def _normalization_features(cls) -> list[dict]:
        return [
            {
                "label"   : "normalization",
                "group"   : "normalization",
                "enable"  : {"normalization.input_strategy": "per_slot", "normalization.output_strategy": "per_slot"},
                "degrade" : {"normalization.input_strategy": "zscore",   "normalization.output_strategy": "zscore"},
            },
            {
                "label"   : "output_clamp",
                "group"   : "normalization",
                "enable"  : {"normalization.clamp_output": True},
                "degrade" : {"normalization.clamp_output": False},
            },
        ]

    @classmethod
    def _augmentation_features(cls) -> list[dict]:
        return [
            {
                "label"   : "augmentation",
                "group"   : "augmentation",
                "enable"  : {f"augmentation.{key}": value for key, value in cls.AUGMENTATION_DEFAULTS},
                "degrade" : {f"augmentation.{key}": 0.0 for key, _ in cls.AUGMENTATION_DEFAULTS},
            },
        ]

    @classmethod
    def _schedule_features(cls) -> list[dict]:
        warmup   = cls.WARMUP_PREFIX
        complete = cls.COMPLETE_PREFIX

        curriculum_enable = {
            "curriculum.enabled"                  : True,
            "curriculum.swap_epoch"               : cls.CURRICULUM_SWAP_EPOCH,
            f"{complete}use_l1_curve"             : False,
            f"{complete}weight_l1_curve"          : 0.0,
            f"{complete}use_param_l1"             : True,
            f"{complete}weight_param_l1"          : 1.0,
            f"{complete}param_matching"           : cls.PARAM_MATCH_FULL,
            f"{complete}use_active_normalization" : True,
            f"{complete}presence_balance"         : False,
        }

        return [
            {
                "label"   : "warmup_loss",
                "group"   : "warmup",
                "enable"  : {
                    f"{warmup}use_l1_curve"             : False,
                    f"{warmup}weight_l1_curve"          : 0.0,
                    f"{warmup}use_param_l1"             : True,
                    f"{warmup}weight_param_l1"          : 1.0,
                    f"{warmup}param_matching"           : cls.PARAM_MATCH_FULL,
                    f"{warmup}use_active_normalization" : True,
                    f"{warmup}presence_balance"         : False,
                },
                "degrade" : {
                    f"{warmup}use_param_l1"             : False,
                    f"{warmup}weight_param_l1"          : 0.0,
                    f"{warmup}use_l1_curve"             : True,
                    f"{warmup}weight_l1_curve"          : 1.0,
                    f"{warmup}param_matching"           : cls.PARAM_MATCH_BASELINE,
                    f"{warmup}use_active_normalization" : False,
                    f"{warmup}presence_balance"         : False,
                },
            },
            {
                "label"   : "lr_warmup",
                "group"   : "schedule",
                "enable"  : {"training.warmup_enabled": True},
                "degrade" : {"training.warmup_enabled": False},
            },
            {
                "label"   : "lr_per_group",
                "group"   : "optimizer",
                "enable"  : {"model_overrides": {key: lr                  for key, lr in cls.GROUP_LR_DEFAULTS}},
                "degrade" : {"model_overrides": {key: cls.SINGLE_GROUP_LR for key, _  in cls.GROUP_LR_DEFAULTS}},
            },
            {
                "label"   : "curriculum",
                "group"   : "schedule",
                "enable"  : curriculum_enable,
                "degrade" : {"curriculum.enabled": False},
            },
        ]

    @classmethod
    def _physics_features(cls) -> list[dict]:
        prefix = cls.COMPLETE_PREFIX

        return [
            {
                "label"   : label,
                "group"   : "physics",
                "enable"  : {f"{prefix}{use_key}": True,  f"{prefix}{weight_key}": weight},
                "degrade" : {f"{prefix}{use_key}": False, f"{prefix}{weight_key}": 0.0},
            }
            for label, use_key, weight_key, weight in cls.PHYSICS_TERMS
        ]

    @classmethod
    def _imbalance_features(cls) -> list[dict]:
        prefix = cls.WARMUP_PREFIX

        return [
            {
                "label"   : "class_imbalance",
                "group"   : "class imbalance",
                "enable"  : {f"{prefix}use_active_normalization": True,  f"{prefix}amp_focal_gamma": 2.0},
                "degrade" : {f"{prefix}use_active_normalization": False, f"{prefix}amp_focal_gamma": 0.0},
            },
        ]

    @classmethod
    def _slot_features(cls) -> list[dict]:
        prefix = cls.WARMUP_PREFIX

        return [
            {
                "label"   : "predict_presence",
                "group"   : "slot presence",
                "enable"  : {"predict_presence": True,  f"{prefix}use_presence_bce": True,  f"{prefix}weight_presence_bce": 1.0},
                "degrade" : {"predict_presence": False, f"{prefix}use_presence_bce": False, f"{prefix}weight_presence_bce": 0.0},
            },
            {
                "label"   : "focal",
                "group"   : "slot presence",
                "enable"  : {f"{prefix}amp_focal_gamma": 2.0},
                "degrade" : {f"{prefix}amp_focal_gamma": 0.0},
            },
            {
                "label"   : "active_norm",
                "group"   : "slot presence",
                "enable"  : {f"{prefix}use_active_normalization": True},
                "degrade" : {f"{prefix}use_active_normalization": False},
            },
            {
                "label"   : "presence_balance",
                "group"   : "slot presence",
                "enable"  : {f"{prefix}presence_balance": True},
                "degrade" : {f"{prefix}presence_balance": False},
            },
        ]

    @classmethod
    def _structural_features(cls) -> list[dict]:
        prefix = cls.WARMUP_PREFIX

        return [
            {
                "label"   : "curve_loss_mse_to_l1",
                "group"   : "loss swap",
                "enable"  : {f"{prefix}use_mse_curve": True,  f"{prefix}weight_mse_curve": 1.0, f"{prefix}use_l1_curve": False, f"{prefix}weight_l1_curve": 0.0},
                "degrade" : {f"{prefix}use_mse_curve": False, f"{prefix}weight_mse_curve": 0.0, f"{prefix}use_l1_curve": True,  f"{prefix}weight_l1_curve": 1.0},
            },
        ]

    @classmethod
    def _architecture_features(cls) -> list[dict]:
        return [
            {
                "label"   : "architecture",
                "group"   : "architecture",
                "enable"  : {"backbone_name": cls.FULL_ARCHITECTURE},
                "degrade" : {"backbone_name": cls.BASELINE_ARCHITECTURE},
            },
        ]

    @classmethod
    def _loss_toggle(cls, label: str, use_key: str, weight_key: str, weight: float) -> dict:
        prefix = cls.WARMUP_PREFIX

        return {
            "label"   : label,
            "group"   : "loss terms",
            "enable"  : {f"{prefix}{use_key}": True,  f"{prefix}{weight_key}": weight},
            "degrade" : {f"{prefix}{use_key}": False, f"{prefix}{weight_key}": 0.0},
        }

    @classmethod
    def features(cls) -> list[dict]:
        loss_terms = [cls._loss_toggle(label, use_key, weight_key, weight) for label, use_key, weight_key, weight in cls.LOSS_TERMS]

        return (
            cls._channel_norm_features()
            + cls._normalization_features()
            + cls._augmentation_features()
            + cls._schedule_features()
            + cls._physics_features()
            + cls._imbalance_features()
            + cls._slot_features()
            + cls._structural_features()
            + cls._architecture_features()
            + loss_terms
        )

    @classmethod
    def as_dict(cls) -> dict:
        return {feature["label"]: feature for feature in cls.features()}

    DEFAULT_ORDER = (
        "covariance_match", "coherence_resyn",
        "curriculum", "augmentation", "architecture",
        "warmup_loss", "lr_per_group", "lr_warmup",
        "out_sigma", "out_amp", "ifg_phase", "pass_mag",
        "output_clamp",
    )

    @classmethod
    def default_features(cls) -> list[dict]:
        catalog = cls.as_dict()
        return [catalog[label] for label in cls.DEFAULT_ORDER]
