from __future__ import annotations

from models import BACKBONE_CONFIG_REGISTRY


class ModelBuilder:
    @staticmethod
    def model_key(model_name: str, head: str) -> str:
        return model_name if head == "conv" else f"{model_name}-{head}"

    @staticmethod
    def split_key(model_key: str) -> tuple[str, str]:
        name, sep, head = model_key.partition("-")
        return (name, head) if sep else (name, "conv")

    @staticmethod
    def config_from_registry(model_name: str, model_overrides: dict, head: str | None = None, registry: dict | None = None):
        registry = BACKBONE_CONFIG_REGISTRY if registry is None else registry
        config   = registry[model_name]()

        if head is not None:
            if "head" in model_overrides:
                raise ValueError("Select the head via the dedicated head field, not model_overrides['head']")
            model_overrides = {"head": head, **model_overrides}

        for attribute, value in model_overrides.items():
            if not hasattr(config, attribute):
                raise AttributeError(f"Unknown model override '{attribute}' for {type(config).__name__}")
            setattr(config, attribute, value)

        return config
