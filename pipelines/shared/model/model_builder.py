from __future__ import annotations

from models import BACKBONE_CONFIG_REGISTRY


class ModelBuilder:
    @staticmethod
    def config_from_registry(model_name: str, model_overrides: dict, registry: dict | None = None):
        registry = BACKBONE_CONFIG_REGISTRY if registry is None else registry
        config   = registry[model_name]()

        for attribute, value in model_overrides.items():
            setattr(config, attribute, value)

        return config
