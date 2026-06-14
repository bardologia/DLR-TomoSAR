from configuration.model.autoencoder_models_config import (
    AutoencoderBaseConfig,
    Conv1dAutoencoderConfig,
    MlpAutoencoderConfig,
    Transformer1dAutoencoderConfig,
)
from .base import AutoencoderBase, AutoencoderBlocks
from .mlp import MlpAutoencoder
from .conv1d import Conv1dAutoencoder
from .transformer1d import Transformer1dAutoencoder


AE_MODEL_REGISTRY: dict[str, type] = {
    "mlp_ae"           : MlpAutoencoder,
    "conv1d_ae"        : Conv1dAutoencoder,
    "transformer1d_ae" : Transformer1dAutoencoder,
}

AE_CONFIG_REGISTRY: dict[str, type] = {
    "mlp_ae"           : MlpAutoencoderConfig,
    "conv1d_ae"        : Conv1dAutoencoderConfig,
    "transformer1d_ae" : Transformer1dAutoencoderConfig,
}


def get_autoencoder(name: str, config=None, **overrides):
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in AE_MODEL_REGISTRY:
        raise ValueError(f"Unknown autoencoder '{name}'. Available: {list(AE_MODEL_REGISTRY.keys())}")
    if config is None:
        config = AE_CONFIG_REGISTRY[key](**overrides)
    elif overrides:
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return AE_MODEL_REGISTRY[key](config), config


__all__ = [
    "AutoencoderBase",
    "AutoencoderBlocks",
    "MlpAutoencoder",
    "Conv1dAutoencoder",
    "Transformer1dAutoencoder",
    "AutoencoderBaseConfig",
    "MlpAutoencoderConfig",
    "Conv1dAutoencoderConfig",
    "Transformer1dAutoencoderConfig",
    "AE_MODEL_REGISTRY",
    "AE_CONFIG_REGISTRY",
    "get_autoencoder",
]
