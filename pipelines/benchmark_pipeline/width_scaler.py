from __future__ import annotations

from dataclasses import dataclass

from models import CONFIG_REGISTRY


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
