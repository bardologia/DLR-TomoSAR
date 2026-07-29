from __future__ import annotations

from dataclasses import fields

from models import BACKBONE_CONFIG_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS


class ModelBuilder:

    ALL_GROUPS_LR = "all_groups_lr"

    @staticmethod
    def model_key(model_name: str, head: str) -> str:
        return model_name if head == "conv" else f"{model_name}-{head}"

    @staticmethod
    def image_size_override(model_name: str, patch_size) -> dict:
        if model_name not in BACKBONE_IMAGE_SIZE_MODELS:
            return {}

        height, width = patch_size
        if height != width:
            raise ValueError(f"Backbone '{model_name}' tokenizes from a single square image_size and cannot run on rectangular patches; got patch_size={tuple(patch_size)}. Use a square patch or a fully-convolutional backbone.")

        return {"image_size": int(height)}

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

        if ModelBuilder.ALL_GROUPS_LR in model_overrides:
            model_overrides = dict(model_overrides)
            uniform_lr      = model_overrides.pop(ModelBuilder.ALL_GROUPS_LR)
            lr_fields       = [spec.name for spec in fields(config) if spec.name.endswith("_lr")]

            if not lr_fields:
                raise AttributeError(f"Model override '{ModelBuilder.ALL_GROUPS_LR}' found no *_lr fields on {type(config).__name__}")

            for name in lr_fields:
                setattr(config, name, uniform_lr)

        for attribute, value in model_overrides.items():
            if not hasattr(config, attribute):
                raise AttributeError(f"Unknown model override '{attribute}' for {type(config).__name__}")
            setattr(config, attribute, value)

        return config
