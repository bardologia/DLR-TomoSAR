from __future__ import annotations

import gc
from dataclasses import dataclass, field

from configuration.benchmark_config import BenchmarkConfig
from configuration.training_config  import GaussianConfig
from models import CONFIG_REGISTRY, get_model
from tools.logger import Logger

_IMAGE_SIZE_MODELS = {"swin_unet", "transunet", "unetr"}


@dataclass
class WidthRule:
    attribute : str
    divisor   : int


class WidthScaler:
    def __init__(self) -> None:
        feature_rules = [WidthRule(attribute="features", divisor=8)]

        self.rules : dict[str, list[WidthRule]] = {
            "unet"                : feature_rules,
            "unet_multihead"      : feature_rules,
            "unet_pergaussian"    : feature_rules,
            "unet_skip"           : feature_rules,
            "resunet"             : feature_rules,
            "resunet_multihead"   : feature_rules,
            "resunet_pergaussian" : feature_rules,
            "attention_unet"      : feature_rules,
            "unetplusplus"        : feature_rules,
            "linknet"             : feature_rules,
            "swin_unet"           : [WidthRule(attribute="embedding_dim", divisor=10)],
            "transunet"           : [WidthRule(attribute="cnn_features",  divisor=8)],
            "unetr"               : [WidthRule(attribute="embedding_dim", divisor=8), WidthRule(attribute="decoder_features", divisor=8)],
            "deeplabv3plus"       : feature_rules,
            "segformer"           : [WidthRule(attribute="embedding_dims", divisor=8), WidthRule(attribute="decoder_channels", divisor=8)],
            "convnext_unet"       : feature_rules,
            "dense_unet"          : [WidthRule(attribute="growth_rate", divisor=2)],
            "hrnet"               : [WidthRule(attribute="base_channels", divisor=8)],
            "multires_unet"       : feature_rules,
            "fpn"                 : feature_rules + [WidthRule(attribute="pyramid_channels", divisor=8)],
            "u2net"               : feature_rules,
        }

    def overrides(self, model_name: str, scale: float) -> dict:
        if model_name not in self.rules:
            raise ValueError(f"No width rule registered for model '{model_name}'. Available: {list(self.rules.keys())}")

        defaults  = CONFIG_REGISTRY[model_name]()
        overrides = {}

        for rule in self.rules[model_name]:
            base = getattr(defaults, rule.attribute)

            if isinstance(base, list):
                overrides[rule.attribute] = [self._round(value, scale, rule.divisor) for value in base]
            else:
                overrides[rule.attribute] = self._round(base, scale, rule.divisor)

        return overrides

    def scaled_config(self, model_name: str, scale: float):
        config = CONFIG_REGISTRY[model_name]()

        for attribute, value in self.overrides(model_name, scale).items():
            setattr(config, attribute, value)

        return config

    def _round(self, value: int, scale: float, divisor: int) -> int:
        scaled = int(round(value * scale / divisor)) * divisor
        return max(divisor, scaled)


@dataclass
class SizeMatchResult:
    model         : str
    scale         : float
    overrides     : dict
    parameters    : int
    target        : int
    deviation_pct : float
    iterations    : int
    history       : list[dict] = field(default_factory=list)


class SizeMatcher:
    def __init__(self, config: BenchmarkConfig, logger: Logger) -> None:
        self.config       = config
        self.logger       = logger
        self.scaler       = WidthScaler()
        self.image_size   = config.training.patch_size[0]
        self.in_channels  = config.size_match.in_channels

        gaussian_cfg      = GaussianConfig.from_dataset(config.paths.dataset_path, n_gaussians=config.n_gaussians)
        self.out_channels = gaussian_cfg.params_per_gaussian * config.n_gaussians

    def reference_count(self) -> int:
        reference = self.config.size_match.reference_model
        return self._count(reference, CONFIG_REGISTRY[reference]())

    def match(self, model_name: str, target: int) -> SizeMatchResult:
        size_match = self.config.size_match

        low  = size_match.scale_low
        high = size_match.scale_high
        while self._count_at(model_name, high) < target and high < 64.0:
            high *= 2.0

        best    : tuple | None = None
        history : list[dict]   = []

        for iteration in range(1, size_match.max_iterations + 1):
            scale      = (low * high) ** 0.5
            parameters = self._count_at(model_name, scale)
            deviation  = (parameters - target) / max(target, 1)

            history.append({"iteration": iteration, "scale": scale, "parameters": parameters, "deviation_pct": 100.0 * deviation})

            if best is None or abs(deviation) < abs(best[1]):
                best = (scale, deviation, parameters)

            if abs(deviation) <= size_match.tolerance:
                break

            if parameters < target:
                low = scale
            else:
                high = scale

        scale, deviation, parameters = best

        return SizeMatchResult(
            model         = model_name,
            scale         = scale,
            overrides     = self.scaler.overrides(model_name, scale),
            parameters    = parameters,
            target        = target,
            deviation_pct = 100.0 * deviation,
            iterations    = len(history),
            history       = history,
        )

    def _count_at(self, model_name: str, scale: float) -> int:
        return self._count(model_name, self.scaler.scaled_config(model_name, scale))

    def _count(self, model_name: str, model_config) -> int:
        overrides = {"in_channels": self.in_channels, "out_channels": self.out_channels}
        if model_name in _IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model, _   = get_model(model_name, config=model_config, **overrides)
        parameters = sum(p.numel() for p in model.parameters())

        del model
        gc.collect()

        return parameters
