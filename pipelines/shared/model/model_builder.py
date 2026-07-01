from __future__ import annotations

from models import BACKBONE_CONFIG_REGISTRY


class ModelBuilder:
    @staticmethod
    def config_from_registry(backbone_name: str, model_overrides: dict):
        config = BACKBONE_CONFIG_REGISTRY[backbone_name]()

        for attribute, value in model_overrides.items():
            setattr(config, attribute, value)

        return config
